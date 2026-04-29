"""
MoNa-pi Training — PaliGemma-3B backbone + Flow Matching head
Usage:
    .venv/bin/python training/train.py
    .venv/bin/python training/train.py --data_dir /path/to/dataset
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, random_split
from accelerate import Accelerator
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones.paligemma_backbone import PaliGemmaBackbone
from models.pi0_core import Pi0VLA
from data.dataset import ActionChunkDataset


# ── Hyperparameters ──────────────────────────────────────────────────────────
DEFAULTS = dict(
    data_dir   = "/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5",
    epochs     = 10,
    batch_size = 2,       # GB10 128GB 통합 메모리 기준 (VLM 포함)
    lr         = 2e-5,    # fine-tuning용 낮은 LR
    warmup     = 100,     # steps
    grad_clip  = 1.0,
    val_ratio  = 0.1,
    num_workers= 0,       # GB10 통합 메모리 — 멀티프로세스 I/O 경합 방지
    ckpt_dir   = "checkpoints",
    save_every = 5,       # epoch마다 저장
    log_every  = 10,      # step마다 로그
)
# ─────────────────────────────────────────────────────────────────────────────


def get_cosine_lr(optimizer, step: int, warmup: int, total: int, base_lr: float):
    import math
    if step < warmup:
        lr = base_lr * step / max(1, warmup)
    else:
        progress = (step - warmup) / max(1, total - warmup)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def build_model(dtype):
    backbone = PaliGemmaBackbone(
        load_pretrained=True,
        freeze_vision=True,
        freeze_language=False,
        dtype=dtype,
    )
    model = Pi0VLA(
        backbone=backbone,
        action_dim=3,
        horizon=10,
        hidden_dim=512,
        backbone_out_dim=backbone.out_dim,
    )
    return model


def train(cfg: dict):
    # BF16: FP16보다 range가 넓어 오버플로우 없음, GB10 Blackwell 네이티브 지원
    # GradScaler 불필요 → clip_grad_norm_ 정상 동작
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    dtype  = torch.bfloat16

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"  MoNa-pi Training")
        print(f"  data : {cfg['data_dir']}")
        print(f"  device: {device}  |  dtype: {dtype}")
        print(f"{'='*60}\n")

    # ── Dataset ──────────────────────────────────────────────────
    full_ds = ActionChunkDataset(
        directory=cfg['data_dir'],
        k=10, window_size=8,
        augment=True,
    )
    n_val  = max(1, int(len(full_ds) * cfg['val_ratio']))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_ds, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=cfg['num_workers'], pin_memory=False,  # 통합 메모리 — pin 불필요
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['num_workers'], pin_memory=False,
    )

    if accelerator.is_main_process:
        print(f"Dataset: {len(full_ds)} samples (train={n_train}, val={n_val})")

    # ── Model ─────────────────────────────────────────────────────
    model = build_model(dtype)

    total_params    = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    if accelerator.is_main_process:
        print(f"Model: {total_params:.1f}M total, {trainable_params:.1f}M trainable\n")

    # ── Optimizer ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['lr'], weight_decay=0.01,
    )

    # ── Accelerate prepare ────────────────────────────────────────
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    total_steps = cfg['epochs'] * len(train_loader)
    global_step = 0
    best_val_loss = float('inf')

    os.makedirs(cfg['ckpt_dir'], exist_ok=True)

    # ── Training Loop ─────────────────────────────────────────────
    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{cfg['epochs']}",
                    disable=not accelerator.is_main_process)

        for batch in pbar:
            images       = batch['images']            # (B, W, 3, 224, 224)
            actions      = batch['actions']           # (B, 10, 3)
            instructions = list(batch['instructions'])

            lr = get_cosine_lr(optimizer, global_step, cfg['warmup'], total_steps, cfg['lr'])

            optimizer.zero_grad()
            loss = model.compute_loss(images, instructions, actions)
            accelerator.backward(loss)

            if cfg['grad_clip'] > 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg['grad_clip'])

            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            global_step += 1

            if global_step % cfg['log_every'] == 0 and accelerator.is_main_process:
                pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}")

        # ── Validation ────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images       = batch['images']
                actions      = batch['actions']
                instructions = list(batch['instructions'])
                val_loss += model.compute_loss(images, instructions, actions).item()
        val_loss /= max(1, len(val_loader))
        epoch_loss /= max(1, len(train_loader))

        elapsed = time.time() - t0
        if accelerator.is_main_process:
            print(f"  Epoch {epoch+1:02d} | train={epoch_loss:.4f} | val={val_loss:.4f} | "
                  f"lr={lr:.2e} | {elapsed:.0f}s")

        # ── Checkpoint ────────────────────────────────────────────
        if accelerator.is_main_process:
            if (epoch + 1) % cfg['save_every'] == 0:
                ckpt = os.path.join(cfg['ckpt_dir'], f"mona_pi_epoch_{epoch+1:02d}")
                accelerator.save_model(model, ckpt)
                print(f"  Saved: {ckpt}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = os.path.join(cfg['ckpt_dir'], "mona_pi_best")
                accelerator.save_model(model, ckpt)
                print(f"  Best model updated (val={val_loss:.4f})")

    if accelerator.is_main_process:
        print("\nTraining complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    train(vars(args))
