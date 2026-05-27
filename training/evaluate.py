"""
MoNa-pi 오프라인 평가 스크립트

메트릭:
    FPE      — mean ||pred - gt||₂  (raw 물리 공간, 발표 ablation 기준)
    Raw MSE  — MSE in physical units (m/s, rad/s)
    Per-dim  — lx / ly / az 분해

실행:
    python training/evaluate.py --config configs/train.yaml --ckpt checkpoints/best
    python training/evaluate.py --config configs/train.yaml --ckpt checkpoints/epoch_023_loss_0.0740
    python training/evaluate.py --config configs/train.yaml --ckpt checkpoints/best --split train
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pi0_core import Pi0VLA
from data.dataset import build_train_val_split


def load_model(cfg: dict, ckpt_path: Path, device: torch.device) -> Pi0VLA:
    model = Pi0VLA(
        action_dim=cfg["model"]["action_dim"],
        horizon=cfg["model"]["horizon"],
        hidden_dim=cfg["model"]["hidden_dim"],
        use_paligemma=cfg["model"].get("use_paligemma", True),
        load_pretrained_paligemma=cfg["model"].get("load_pretrained_paligemma", False),
        paligemma_id=cfg["model"].get("paligemma_id", "google/paligemma-3b-pt-224"),
        vision_model_id=cfg["model"].get("vision_model_id", "google/siglip-so400m-patch14-384"),
        lang_model_id=cfg["model"].get("lang_model_id", "google/gemma-2b"),
    )

    sf = ckpt_path / "model.safetensors"
    pt = ckpt_path / "pytorch_model.bin"

    if sf.exists():
        state = load_file(str(sf), device="cpu")
        model.load_state_dict(state, strict=True)
        print(f"[Eval] 가중치 로드: {sf}")
    elif pt.exists():
        state = torch.load(pt, map_location="cpu")
        model.load_state_dict(state, strict=True)
        print(f"[Eval] 가중치 로드: {pt}")
    else:
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_path}")

    return model.to(device).eval()


@torch.no_grad()
def evaluate(model: Pi0VLA, loader: DataLoader, device: torch.device) -> dict:
    fpe_sum      = 0.0   # L2 distance (raw)
    raw_mse_sum  = 0.0
    per_dim_sum  = np.zeros(3)
    n_samples    = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        images       = batch["images"].to(device)        # (B, N, C, H, W)
        actions_gt   = batch["actions"].to(device)       # (B, k, 3) raw physical
        instructions = batch["instructions"]

        # sample_actions → raw physical space (unnormalized by MoNaActionExpert)
        actions_pred = model.sample_actions(images, instructions, n_steps=5)  # (B, k, 3)

        diff = (actions_pred - actions_gt).float()       # (B, k, 3)

        # FPE: mean L2 norm per sample, averaged across horizon
        fpe  = diff.norm(dim=-1).mean(dim=-1)            # (B,)
        fpe_sum     += fpe.sum().item()

        # Raw MSE
        raw_mse_sum += diff.pow(2).mean().item() * images.shape[0]

        # Per-dim MSE
        per_dim_sum += diff.pow(2).reshape(-1, 3).mean(dim=0).cpu().numpy() * images.shape[0]

        n_samples += images.shape[0]

    return {
        "fpe":     fpe_sum / n_samples,
        "raw_mse": raw_mse_sum / n_samples,
        "per_dim_mse": {
            "linear_x":  float(per_dim_sum[0] / n_samples),
            "linear_y":  float(per_dim_sum[1] / n_samples),
            "angular_z": float(per_dim_sum[2] / n_samples),
        },
        "n_samples": n_samples,
    }


def print_results(results: dict, ckpt: str, split: str):
    print()
    print("=" * 52)
    print("  MoNa-pi 오프라인 평가 결과")
    print(f"  ckpt  : {ckpt}")
    print(f"  split : {split}  (n={results['n_samples']})")
    print("=" * 52)
    print(f"  FPE      : {results['fpe']:.4f}  ← 발표 ablation 기준")
    print(f"  Raw MSE  : {results['raw_mse']:.6f}")
    print("  Per-Dim  :")
    for k, v in results["per_dim_mse"].items():
        print(f"    {k:<12}: {v:.6f}")
    print("=" * 52)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--ckpt",   required=True)
    parser.add_argument("--split",  default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train"].get("seed", 42))

    # val split — training과 동일한 seed 사용
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

    ds = val_ds if args.split == "val" else train_ds
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = load_model(cfg, Path(args.ckpt), device)
    results = evaluate(model, loader, device)
    print_results(results, args.ckpt, args.split)


if __name__ == "__main__":
    main()
