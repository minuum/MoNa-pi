"""
lerobot/pi0_old에서 PaliGemma backbone 가중치 추출
→ MoNa-pi backbone 형식으로 리매핑 후 저장

lerobot/pi0_old: robot demonstration data로 추가 학습된 PaliGemma
  model.paligemma_with_expert.paligemma.model.vision_tower.*
  model.paligemma_with_expert.paligemma.model.multi_modal_projector.*
  model.paligemma_with_expert.paligemma.model.language_model.*

MoNa-pi:
  backbone.vision_tower.*
  backbone.projector.*
  backbone.language_model.*

사용:
    python scripts/extract_lerobot_backbone.py
    → checkpoints/lerobot_pi0_backbone/ 에 저장
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print("[1/3] lerobot/pi0_old 다운로드 중...")
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
import torch

ckpt_path = hf_hub_download("lerobot/pi0_old", "model.safetensors")
print(f"  다운로드 완료: {ckpt_path}")

print("[2/3] backbone 가중치 추출 및 리매핑...")

PREFIX = "model.paligemma_with_expert.paligemma."
# lerobot vision_tower는 .vision_model. 중간 레이어 포함 → 제거
REMAP = {
    "model.vision_tower.vision_model.": "backbone.vision_tower.",  # vision_model 래퍼 제거
    "model.vision_tower.":              "backbone.vision_tower.",  # fallback (head 등)
    "model.multi_modal_projector.":     "backbone.projector.",
    "model.language_model.":            "backbone.language_model.",
}

remapped = {}
skipped = []

with safe_open(ckpt_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        if not key.startswith(PREFIX):
            skipped.append(key)
            continue
        # PREFIX 제거
        sub = key[len(PREFIX):]  # e.g. "model.vision_tower.xxx"
        # REMAP 적용
        new_key = None
        for old_prefix, new_prefix in REMAP.items():
            if sub.startswith(old_prefix):
                new_key = new_prefix + sub[len(old_prefix):]
                break
        if new_key is None:
            skipped.append(key)
            continue
        remapped[new_key] = f.get_tensor(key)

print(f"  매핑됨: {len(remapped)}개")
print(f"  건너뜀: {len(skipped)}개 ({', '.join(skipped[:3])}...)")

print("[3/4] backbone 저장 중...")
out_dir = Path("checkpoints/lerobot_pi0_backbone")
out_dir.mkdir(parents=True, exist_ok=True)
save_file(remapped, str(out_dir / "model.safetensors"))
print(f"  저장 완료: {out_dir}/model.safetensors")
print(f"  총 {len(remapped)}개 레이어 (vision_tower + projector + language_model)")

# ── gemma_expert 별도 추출 ───────────────────────────────────────────────
print("[4/4] gemma_expert 가중치 추출...")
EXPERT_PREFIX = "model.paligemma_with_expert.gemma_expert."
expert = {}
with safe_open(ckpt_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        if key.startswith(EXPERT_PREFIX):
            sub = key[len(EXPERT_PREFIX):]   # e.g. "model.layers.0.*" or "lm_head.*"
            # GemmaModel state dict: .model. 레벨 제거
            if sub.startswith("model."):
                sub = sub[len("model."):]    # "layers.0.*"
            new_key = "gemma." + sub
            expert[new_key] = f.get_tensor(key)

expert_dir = Path("checkpoints/lerobot_gemma_expert")
expert_dir.mkdir(parents=True, exist_ok=True)
save_file(expert, str(expert_dir / "model.safetensors"))
print(f"  저장 완료: {expert_dir}/model.safetensors")
print(f"  총 {len(expert)}개 레이어 (18-layer Gemma expert, hidden=1024)")
