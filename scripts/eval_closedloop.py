"""
MoNa-pi Offline Closed-Loop 시뮬레이터
MoNaVLA rollout_core.py 방식을 continuous 3-DOF로 확장.

각 에피소드를 순서대로 실행:
  t=0: GT 이미지 → 모델 → action[0] → pose 업데이트
  t=1: GT 이미지 → 모델 → action[0] → pose 업데이트
  ...
  → 예측 궤적 vs GT 궤적 비교 → FPE, TLD, Success

실행:
    python scripts/eval_closedloop.py --config configs/train.yaml --ckpt checkpoints/best
    python scripts/eval_closedloop.py --config configs/train.yaml --ckpt checkpoints/best --n-eps 20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.pi0_core import Pi0VLA
from data.dataset import ActionChunkDataset, build_train_val_split
from data.preprocessing import IntentPrefixInjector
from safetensors.torch import load_file

# 평가 시 사용할 intent injector (is_training=False → 노이즈 없음)
_INJECTOR = IntentPrefixInjector()


# ── 운동학 (MoNaVLA rollout_core 직접 이관, continuous) ─────────────────────

@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


def pose_step(pose: Pose, lx: float, ly: float, az: float, dt: float = 0.1) -> Pose:
    """Body-frame velocity → world-frame pose update (옴니휠 운동학)."""
    ct, st = math.cos(pose.theta), math.sin(pose.theta)
    return Pose(
        x=pose.x + (lx * ct - ly * st) * dt,
        y=pose.y + (lx * st + ly * ct) * dt,
        theta=pose.theta + az * dt,
    )


def build_trajectory(actions: np.ndarray, dt: float = 0.1) -> List[Pose]:
    """(T, 3) 연속 액션 시퀀스 → Pose 궤적."""
    poses = [Pose()]
    for lx, ly, az in actions:
        poses.append(pose_step(poses[-1], float(lx), float(ly), float(az), dt))
    return poses


def trajectory_length(poses: List[Pose]) -> float:
    total = 0.0
    for i in range(1, len(poses)):
        dx = poses[i].x - poses[i-1].x
        dy = poses[i].y - poses[i-1].y
        total += math.sqrt(dx*dx + dy*dy)
    return total


def compute_metrics(
    expert_poses: List[Pose],
    pred_poses: List[Pose],
    fpe_thresh: float = 0.5,
) -> dict:
    """FPE, TLD, mean lateral deviation, success."""
    n = min(len(expert_poses), len(pred_poses))
    expert_final = np.array([expert_poses[-1].x, expert_poses[-1].y])
    pred_final   = np.array([pred_poses[-1].x,   pred_poses[-1].y])
    fpe = float(np.linalg.norm(pred_final - expert_final))

    expert_len = trajectory_length(expert_poses)
    pred_len   = trajectory_length(pred_poses)
    tld = pred_len / max(expert_len, 1e-6)

    devs = []
    for i in range(n):
        dx = pred_poses[i].x - expert_poses[i].x
        dy = pred_poses[i].y - expert_poses[i].y
        devs.append(math.sqrt(dx*dx + dy*dy))
    mean_dev = float(np.mean(devs)) if devs else 0.0

    success = (fpe < fpe_thresh) and (0.7 <= tld <= 1.5)
    return {
        "fpe": round(fpe, 4),
        "tld": round(float(tld), 4),
        "mean_lateral_dev": round(mean_dev, 4),
        "expert_len": round(float(expert_len), 4),
        "pred_len":   round(float(pred_len),   4),
        "success": bool(success),
        "n_frames": n,
    }


# ── 모델 로드 ────────────────────────────────────────────────────────────────

def load_model(cfg: dict, ckpt: Path, device: torch.device) -> Pi0VLA:
    m = cfg["model"]
    model = Pi0VLA(
        action_dim=m["action_dim"],
        horizon=m["horizon"],
        hidden_dim=m["hidden_dim"],
        use_paligemma=m.get("use_paligemma", True),
        load_pretrained_paligemma=m.get("load_pretrained_paligemma", False),
        paligemma_id=m.get("paligemma_id", "google/paligemma-3b-pt-224"),
        vision_model_id=m.get("vision_model_id", "google/siglip-so400m-patch14-384"),
        lang_model_id=m.get("lang_model_id", "google/gemma-2b"),
        use_gemma_expert=m.get("use_gemma_expert", False),
        use_lora=m.get("use_lora", False),
        lora_r=m.get("lora_r", 16),
    )
    sf = ckpt / "model.safetensors"
    pt = ckpt / "pytorch_model.bin"
    if sf.exists():
        state = load_file(str(sf), device="cpu")
    elif pt.exists():
        state = torch.load(pt, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"checkpoint 없음: {ckpt}")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


# ── 에피소드 단위 CL 평가 ────────────────────────────────────────────────────

@torch.no_grad()
def eval_episode(
    model: Pi0VLA,
    ep_data: dict,
    cfg: dict,
    device: torch.device,
    dt: float = 0.1,
) -> dict:
    """
    단일 에피소드 closed-loop 평가.
    GT 이미지를 순서대로 모델에 입력, 첫 번째 예측 액션으로 궤적 구성.
    """
    images_np   = ep_data["images"]   # (T, H, W, 3) uint8
    gt_actions  = ep_data["actions"]  # (T, 3) float32 raw
    raw_instr   = ep_data["instruction"]  # HDF5 raw (no [INTENT] tag)

    cfg_data  = cfg["data"]
    cfg_model = cfg["model"]
    img_size  = cfg_data.get("image_size", 224)
    window    = cfg_data.get("window_size", 8)
    horizon   = cfg_model["horizon"]

    import torchvision.transforms.functional as TF
    from PIL import Image as PILImage

    pred_actions = []
    T = images_np.shape[0]

    for t in range(window - 1, T - horizon + 1):
        # 이미지 윈도우 준비 (B=1, N=window, C, H, W)
        frames = []
        for i in range(t - window + 1, t + 1):
            pil = PILImage.fromarray(images_np[i]).resize((img_size, img_size))
            arr = np.array(pil).transpose(2, 0, 1).astype(np.float32) / 255.0
            frames.append(torch.from_numpy(arr))
        img_tensor = torch.stack(frames).unsqueeze(0).to(device)  # (1, N, C, H, W)

        # 현재 GT 액션 청크로 intent 태그 주입 (학습 시와 동일 형식)
        t_end   = min(t + horizon, len(gt_actions))
        chunk_gt = gt_actions[t: t_end]
        tagged_instr = _INJECTOR.inject(raw_instr, chunk_gt, is_training=False)

        # 모델 추론: sample_actions → (1, horizon, 3) raw physical
        chunk = model.sample_actions(img_tensor, [tagged_instr], n_steps=5)  # (1,h,3)
        first_action = chunk[0, 0].float().cpu().numpy()  # (3,) [vx, vy, wz]
        pred_actions.append(first_action)

    if not pred_actions:
        return {"skipped": True}

    pred_actions = np.array(pred_actions)  # (T', 3)
    gt_slice     = gt_actions[window - 1: window - 1 + len(pred_actions)]  # (T', 3)

    expert_poses = build_trajectory(gt_slice,   dt)
    pred_poses   = build_trajectory(pred_actions, dt)

    metrics = compute_metrics(expert_poses, pred_poses)
    metrics["instruction"] = raw_instr
    return metrics


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MoNa-pi Offline Closed-Loop 평가")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--ckpt",   default="checkpoints/best")
    parser.add_argument("--split",  default="val", choices=["val", "train"])
    parser.add_argument("--n-eps",  type=int, default=None, help="평가할 에피소드 수 제한")
    parser.add_argument("--fpe-thresh", type=float, default=0.5)
    parser.add_argument("--dt",     type=float, default=0.1, help="제어 주기 (초)")
    parser.add_argument("--out",    default=None, help="결과 JSON 저장 경로")
    parser.add_argument("--regular-only", action="store_true",
                        help="정형 9종 경로만 평가 (center/left/right × straight/left/right)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train"].get("seed", 42))

    # val split — training과 동일 seed
    train_ds, val_ds = build_train_val_split(
        directory=cfg["data"]["train_path"],
        val_split=cfg["data"].get("val_split", 0.1),
        k=cfg["model"]["horizon"],
        window_size=cfg["data"]["window_size"],
        image_size=cfg["data"].get("image_size", 224),
        preprocess=cfg["data"].get("preprocess", True),
        normalize=cfg["data"].get("normalize", False),
        seed=cfg["train"].get("seed", 42),
    )
    base_ds = val_ds.dataset if hasattr(val_ds, "dataset") else val_ds

    # val에 속하는 에피소드 f_idx 추출
    val_indices = val_ds.indices if hasattr(val_ds, "indices") else list(range(len(val_ds)))
    val_f_idxs  = sorted({base_ds.samples[i][0] for i in val_indices})

    # 정형 경로 필터 (--regular-only)
    REGULAR_PATHS = [
        'center_straight', 'center_left', 'center_right',
        'left_straight',   'left_left',   'left_right',
        'right_straight',  'right_left',  'right_right',
    ]
    if args.regular_only:
        h5_files = sorted(Path(cfg["data"]["train_path"]).glob("*.h5"))
        idx_to_stem = {i: f.stem for i, f in enumerate(h5_files)}
        val_f_idxs = [i for i in val_f_idxs
                      if any(kw in idx_to_stem.get(i, '') for kw in REGULAR_PATHS)]
        print(f"[CL Eval] --regular-only: 정형 9종만 ({len(val_f_idxs)}개)")

    if args.n_eps:
        val_f_idxs = val_f_idxs[:args.n_eps]

    print(f"[CL Eval] {args.split} 에피소드 수: {len(val_f_idxs)}")
    print(f"[CL Eval] ckpt: {args.ckpt}, fpe_thresh: {args.fpe_thresh}m")

    model = load_model(cfg, Path(args.ckpt), device)
    print(f"[CL Eval] 모델 로드 완료 (device={device})")

    results = []
    for f_idx in tqdm(val_f_idxs, desc="Episodes"):
        ep = base_ds._ep_cache[f_idx]
        metrics = eval_episode(model, ep, cfg, device, dt=args.dt)
        if metrics.get("skipped"):
            continue
        metrics["episode_idx"] = f_idx
        results.append(metrics)

    if not results:
        print("평가할 에피소드 없음")
        return

    # 집계
    n         = len(results)
    n_ok      = sum(1 for r in results if r["success"])
    mean_fpe  = np.mean([r["fpe"]  for r in results])
    mean_tld  = np.mean([r["tld"]  for r in results])
    mean_dev  = np.mean([r["mean_lateral_dev"] for r in results])

    print()
    print("=" * 54)
    print("  MoNa-pi Offline Closed-Loop 평가 결과")
    print("=" * 54)
    print(f"  에피소드 수   : {n}")
    print(f"  Success Rate  : {n_ok}/{n} = {n_ok/n:.1%}  (FPE<{args.fpe_thresh}m & TLD∈[0.7,1.5])")
    print(f"  Mean FPE      : {mean_fpe:.4f} m")
    print(f"  Mean TLD      : {mean_tld:.4f}")
    print(f"  Mean Lat. Dev : {mean_dev:.4f} m")
    print("=" * 54)

    summary = {
        "ckpt": args.ckpt,
        "split": args.split,
        "n_episodes": n,
        "success_rate": round(n_ok / n, 4),
        "n_success": n_ok,
        "mean_fpe": round(float(mean_fpe), 4),
        "mean_tld": round(float(mean_tld), 4),
        "mean_lateral_dev": round(float(mean_dev), 4),
        "fpe_thresh": args.fpe_thresh,
        "episodes": results,
    }

    out_path = args.out or f"logs/cl_eval_{Path(args.ckpt).name}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  저장: {out_path}")


if __name__ == "__main__":
    main()
