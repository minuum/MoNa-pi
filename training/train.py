"""
MoNa-pi 학습 스크립트

실행:
    python training/train.py --config configs/train.yaml

의존성:
    pip install pyyaml accelerate tqdm torch scipy
    pip install wandb  # use_wandb: true 시
"""

import argparse
import math
import os
import sys
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pi0_core import Pi0VLA
from data.dataset import build_train_val_split


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# 기존 가중치 로드 (하위 호환 유지)
# ─────────────────────────────────────────────

def load_pretrained_weights(model: nn.Module, ckpt_path: str) -> nn.Module:
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[Train] 체크포인트 없음 ({ckpt_path}), 처음부터 학습")
        return model
    print(f"[Train] 가중치 로드: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_dict = model.state_dict()
    matched = {k: v for k, v in checkpoint.items()
               if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(matched)
    model.load_state_dict(model_dict)
    print(f"[Train] {len(matched)}/{len(model_dict)} 레이어 매칭 완료")
    return model


# ─────────────────────────────────────────────
# Warmup + Cosine LR 스케줄러
# ─────────────────────────────────────────────

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────
# 체크포인트 관리 (best top-k 보존)
# ─────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, save_dir: str, keep_top_k: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.records: list[tuple[float, Path]] = []  # (val_loss, path)

    def save(self, accelerator, model, epoch: int, val_loss: float):
        ckpt_path = self.save_dir / f"epoch_{epoch:03d}_loss_{val_loss:.4f}"
        accelerator.save_model(model, str(ckpt_path))
        self.records.append((val_loss, ckpt_path))
        self.records.sort(key=lambda x: x[0])

        # top-k 초과 체크포인트 삭제
        while len(self.records) > self.keep_top_k:
            _, old_path = self.records.pop()
            if old_path.exists():
                shutil.rmtree(old_path, ignore_errors=True)

        is_best = self.records[0][1] == ckpt_path
        if is_best:
            best_path = self.save_dir / "best"
            if best_path.exists():
                shutil.rmtree(best_path, ignore_errors=True)
            shutil.copytree(str(ckpt_path), str(best_path))
        return is_best


# ─────────────────────────────────────────────
# 에포크 단위 학습 / 검증
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, accelerator, grad_clip: float):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train", disable=not accelerator.is_local_main_process, leave=False):
        images = batch["images"]
        actions = batch["actions"]
        instructions = batch["instructions"]

        optimizer.zero_grad()
        loss = model.compute_loss(images, instructions, actions)
        accelerator.backward(loss)

        if grad_clip > 0:
            accelerator.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, accelerator):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Val  ", disable=not accelerator.is_local_main_process, leave=False):
        images = batch["images"]
        actions = batch["actions"]
        instructions = batch["instructions"]
        loss = model.compute_loss(images, instructions, actions)
        total_loss += loss.item()
    return total_loss / len(loader)


# ─────────────────────────────────────────────
# 로거 (W&B or TensorBoard)
# ─────────────────────────────────────────────

def build_logger(cfg: dict):
    log_cfg = cfg.get("logging", {})
    if log_cfg.get("use_wandb", False):
        import wandb
        wandb.init(
            project=log_cfg.get("project", "mona-pi"),
            entity=log_cfg.get("entity", None),
            config=cfg,
        )
        return "wandb"
    else:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = Path(cfg["train"]["checkpoint_dir"]) / "tb_logs"
        writer = SummaryWriter(log_dir=str(log_dir))
        return writer


def log_metrics(logger, metrics: dict, step: int):
    if logger == "wandb":
        import wandb
        wandb.log(metrics, step=step)
    elif hasattr(logger, "add_scalar"):
        for k, v in metrics.items():
            logger.add_scalar(k, v, step)


# ─────────────────────────────────────────────
# 메인 학습 루프
# ─────────────────────────────────────────────

def train(config_path: str):
    cfg = load_config(config_path)
    torch.manual_seed(cfg["train"].get("seed", 42))

    accelerator = Accelerator()
    device = accelerator.device

    # ── 데이터셋 ───────────────────────────────
    train_ds, val_ds = build_train_val_split(
        directory=cfg["data"]["train_path"],
        val_split=cfg["data"].get("val_split", 0.1),
        k=cfg["model"]["horizon"],
        window_size=cfg["data"]["window_size"],
        image_size=cfg["data"].get("image_size", 384),
        preprocess=cfg["data"].get("preprocess", True),
        augment=cfg["data"].get("augment", False),
        use_color_jitter=cfg["data"].get("use_color_jitter", False),
        use_random_crop=cfg["data"].get("use_random_crop", False),
        use_counterfactual=cfg["data"].get("use_counterfactual", False),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ── 모델 ──────────────────────────────────
    model = Pi0VLA(
        action_dim=cfg["model"]["action_dim"],
        horizon=cfg["model"]["horizon"],
        hidden_dim=cfg["model"]["hidden_dim"],
    )
    model = load_pretrained_weights(model, cfg["pretrain"].get("ckpt_path", ""))

    # ── Optimizer & Scheduler ─────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    total_steps = len(train_loader) * cfg["train"]["epochs"]
    scheduler = get_lr_scheduler(optimizer, cfg["train"]["warmup_steps"], total_steps)

    # ── Accelerate 래핑 ────────────────────────
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # ── 로깅 ──────────────────────────────────
    logger = None
    if accelerator.is_local_main_process:
        logger = build_logger(cfg)

    ckpt_mgr = CheckpointManager(
        cfg["train"]["checkpoint_dir"],
        keep_top_k=cfg["train"].get("keep_top_k", 3),
    )

    log_every = cfg.get("logging", {}).get("log_every_n_steps", 20)
    global_step = 0

    # ── 학습 루프 ──────────────────────────────
    for epoch in range(cfg["train"]["epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, accelerator,
            grad_clip=cfg["train"].get("grad_clip", 1.0),
        )
        val_loss = validate(model, val_loader, accelerator)
        global_step += len(train_loader)

        if accelerator.is_local_main_process:
            is_best = ckpt_mgr.save(accelerator, model, epoch, val_loss)
            best_mark = " ★" if is_best else ""
            lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else cfg["train"]["lr"]
            print(
                f"[Epoch {epoch+1:03d}/{cfg['train']['epochs']}] "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"lr={lr_now:.2e}{best_mark}"
            )
            if logger is not None:
                log_metrics(logger, {
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "train/lr": lr_now,
                }, step=global_step)

    if accelerator.is_local_main_process:
        print("[Train] 완료!")
        if logger != "wandb" and hasattr(logger, "close"):
            logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    train(args.config)
