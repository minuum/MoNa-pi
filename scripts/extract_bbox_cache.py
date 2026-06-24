"""
Phase 3 Step A — PaliGemma generate() 기반 "gray basket" bbox 오프라인 추출.

배경: MoNaVLA와 달리 MoNa-pi 데이터(mobile_vla_dataset_merged/)엔 bbox 라벨이
전혀 없다. language_instruction이 거의 전부 "... gray basket ..."으로 타겟이
고정돼 있어, PaliGemma 자체의 "detect {object}" 프롬프트 + generate()로
박스를 뽑아 캐시한다. 오프라인 1회성 스크립트이므로 inference 핫루프와 무관
(MoNaVLA가 경고하는 "실시간 generate() 금지"는 다른 backbone·다른 상황).

사용:
    # 품질 확인 (몇 개 에피소드만, 오버레이 PNG 저장)
    python scripts/extract_bbox_cache.py --probe --n-episodes 3 --out-dir scratch/bbox_probe

    # 전체 캐시 생성
    python scripts/extract_bbox_cache.py --full --out logs/bbox_cache.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_LOC_RE = re.compile(r"<loc(\d{4})>")
PROMPT = "detect gray basket"


def parse_locs(text: str) -> tuple | None:
    """
    PaliGemma detect 출력에서 <locYYYY> 4개(y1,x1,y2,x2, 1024-bin 정규화)를 파싱.
    Returns: (cx, cy, area) normalized [0,1], 또는 박스 없으면 None.
    """
    nums = _LOC_RE.findall(text)
    if len(nums) < 4:
        return None
    y1, x1, y2, x2 = (int(n) / 1024.0 for n in nums[:4])
    if y2 <= y1 or x2 <= x1:
        return None
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    area = (y2 - y1) * (x2 - x1)
    return cx, cy, area, (y1, x1, y2, x2)


def load_paligemma_detector(paligemma_id: str = "google/paligemma-3b-pt-224", device: str = "cuda"):
    from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        paligemma_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(paligemma_id)
    return model, processor


@torch.no_grad()
def detect_frame(model, processor, image_pil, device: str = "cuda", max_new_tokens: int = 20):
    inputs = processor(text=PROMPT, images=image_pil, return_tensors="pt").to(device, torch.bfloat16)
    input_len = inputs["input_ids"].shape[-1]
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    decoded = processor.decode(out[0][input_len:], skip_special_tokens=True)
    return decoded, parse_locs(decoded)


def iter_h5_frames(h5_path: Path, stride: int = 1):
    with h5py.File(h5_path, "r") as f:
        images = f["observations/images"][:]
    for i in range(0, images.shape[0], stride):
        yield i, images[i]


def probe(args):
    """소규모 샘플 검증: 박스를 이미지에 오버레이해 PNG로 저장."""
    from PIL import Image, ImageDraw

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_paligemma_detector(device=device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(Path(args.directory).glob("*.h5"))[: args.n_episodes]
    for f_path in h5_files:
        with h5py.File(f_path, "r") as f:
            images = f["observations/images"][:]
        n = images.shape[0]
        sample_idxs = sorted({0, n // 2, n - 1})
        for i in sample_idxs:
            img = Image.fromarray(images[i]).convert("RGB")
            decoded, parsed = detect_frame(model, processor, img, device=device)
            tag = f"{f_path.stem[:40]}_f{i}"
            print(f"[{tag}] raw='{decoded}' parsed={parsed}")
            draw_img = img.copy()
            if parsed is not None:
                cx, cy, area, (y1, x1, y2, x2) = parsed
                W, H = draw_img.size
                box = (x1 * W, y1 * H, x2 * W, y2 * H)
                d = ImageDraw.Draw(draw_img)
                d.rectangle(box, outline="red", width=4)
                d.text((5, 5), f"cx={cx:.2f} cy={cy:.2f} area={area:.3f}", fill="red")
            else:
                d = ImageDraw.Draw(draw_img)
                d.text((5, 5), f"NO BOX: {decoded}", fill="red")
            draw_img.save(out_dir / f"{tag}.png")
    print(f"\n저장: {out_dir}/ — 이미지를 눈으로 확인해 박스가 바구니를 잡는지 판단할 것.")


def full_extract(args):
    """전체 에피소드/프레임에 대해 bbox 추출, JSON 캐시 저장."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_paligemma_detector(device=device)

    h5_files = sorted(Path(args.directory).glob("*.h5"))
    cache = {}
    for f_path in h5_files:
        try:
            with h5py.File(f_path, "r") as f:
                images = f["observations/images"][:]
        except Exception as e:
            print(f"[건너뜀] {f_path.name}: {e}")
            continue

        from PIL import Image
        frame_results = []
        n_missing = 0
        for i in range(images.shape[0]):
            img = Image.fromarray(images[i]).convert("RGB")
            decoded, parsed = detect_frame(model, processor, img, device=device)
            if parsed is None:
                n_missing += 1
                frame_results.append({"valid": False, "cx": 0.0, "cy": 0.0, "area": 0.0})
            else:
                cx, cy, area, _ = parsed
                frame_results.append({"valid": True, "cx": cx, "cy": cy, "area": area})

        cache[f_path.stem] = frame_results
        print(f"[{f_path.stem}] {images.shape[0]}프레임, 탐지 실패 {n_missing}개")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cache, f)
    print(f"\n저장: {out_path} ({len(cache)}개 에피소드)")


def main():
    ap = argparse.ArgumentParser(description="PaliGemma detect 기반 gray basket bbox 추출")
    ap.add_argument("--directory", default="mobile_vla_dataset_merged")
    ap.add_argument("--probe", action="store_true", help="소규모 품질 확인 모드")
    ap.add_argument("--full", action="store_true", help="전체 캐시 생성 모드")
    ap.add_argument("--n-episodes", type=int, default=3)
    ap.add_argument("--out-dir", default="scratch/bbox_probe")
    ap.add_argument("--out", default="logs/bbox_cache.json")
    args = ap.parse_args()

    if args.probe == args.full:
        raise ValueError("--probe 또는 --full 둘 중 정확히 하나를 지정")

    if args.probe:
        probe(args)
    else:
        full_extract(args)


if __name__ == "__main__":
    main()
