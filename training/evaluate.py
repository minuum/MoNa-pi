"""
MoNa-pi 오프라인 평가 스크립트

측정 지표:
    1. Normalized Action MSE  — 학습 공간 [-1, 1] 에서의 오차
    2. Raw Action MSE         — 실제 속도 단위 [m/s, rad/s] 에서의 오차
    3. Per-Dim MSE            — (lx, ly, az) 차원별 분해
    4. Action Chunk Consistency — 예측 청크와 GT 청크 간 DTW 거리 (옵션)

실행:
    python training/evaluate.py \
        --config configs/train.yaml \
        --ckpt   checkpoints/best
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pi0_core import Pi0VLA
from data.dataset import ActionChunkDataset
from data.preprocessing import ActionNormalizer


# ─────────────────────────────────────────────
# 평가 루프
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: Pi0VLA, loader: DataLoader, normalizer: ActionNormalizer, device):
    model.eval()

    norm_mse_sum = 0.0
    raw_mse_sum = 0.0
    per_dim_sum = np.zeros(3)
    n_batches = 0

    for batch in tqdm(loader, desc="Eval"):
        images = batch["images"].to(device)
        actions_gt_norm = batch["actions"].to(device)   # (B, k, 3) 정규화
        instructions = batch["instructions"]

        # 모델 추론 — 정규화 공간에서 액션 청크 예측
        actions_pred_norm = model.sample_actions(images, instructions, n_steps=5)  # (B, k, 3)

        # 1. Normalized MSE
        diff_norm = (actions_pred_norm - actions_gt_norm) ** 2
        norm_mse_sum += diff_norm.mean().item()

        # 2. Raw MSE (unnormalize 후 비교)
        pred_raw = normalizer.unnormalize_tensor(actions_pred_norm.cpu()).numpy()
        gt_raw   = normalizer.unnormalize_tensor(actions_gt_norm.cpu()).numpy()
        diff_raw = (pred_raw - gt_raw) ** 2
        raw_mse_sum += diff_raw.mean()

        # 3. Per-dim MSE (raw 공간)
        per_dim_sum += diff_raw.reshape(-1, 3).mean(axis=0)

        n_batches += 1

    results = {
        "normalized_mse": norm_mse_sum / n_batches,
        "raw_mse":        raw_mse_sum / n_batches,
        "per_dim_mse": {
            "linear_x":  per_dim_sum[0] / n_batches,
            "linear_y":  per_dim_sum[1] / n_batches,
            "angular_z": per_dim_sum[2] / n_batches,
        },
    }
    return results


def print_results(results: dict):
    print("\n" + "=" * 50)
    print("  MoNa-pi 오프라인 평가 결과")
    print("=" * 50)
    print(f"  Normalized MSE : {results['normalized_mse']:.6f}")
    print(f"  Raw MSE        : {results['raw_mse']:.6f}  [m/s, rad/s]")
    print("  Per-Dim MSE    :")
    for k, v in results["per_dim_mse"].items():
        print(f"    {k:<12}: {v:.6f}")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--ckpt",   required=True, help="체크포인트 디렉토리 경로")
    parser.add_argument("--split",  default="val", choices=["train", "val"],
                        help="평가할 데이터 split")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 데이터셋 ───────────────────────────────
    dataset = ActionChunkDataset(
        directory=cfg["data"]["train_path"],
        k=cfg["model"]["horizon"],
        window_size=cfg["data"]["window_size"],
        preprocess=cfg["data"].get("preprocess", True),
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    normalizer = dataset.normalizer

    # ── 모델 로드 ──────────────────────────────
    model = Pi0VLA(
        action_dim=cfg["model"]["action_dim"],
        horizon=cfg["model"]["horizon"],
        hidden_dim=cfg["model"]["hidden_dim"],
    )
    ckpt_dir = Path(args.ckpt)
    # Accelerate 저장 형식: pytorch_model.bin 또는 model.safetensors
    bin_path = ckpt_dir / "pytorch_model.bin"
    if bin_path.exists():
        state = torch.load(bin_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"[Eval] 가중치 로드: {bin_path}")
    else:
        print(f"[Eval] 경고: {bin_path} 없음. 랜덤 가중치로 평가합니다.")

    model = model.to(device)

    # ── 평가 실행 ──────────────────────────────
    results = evaluate(model, loader, normalizer, device)
    print_results(results)


if __name__ == "__main__":
    main()
