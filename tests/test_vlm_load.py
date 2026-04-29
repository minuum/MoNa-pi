"""
VLM 로드 & 추론 파이프라인 테스트.
사전학습 가중치 없이도 --mock 으로 구조만 검증 가능.

Usage:
    python tests/test_vlm_load.py           # 실제 가중치 로드
    python tests/test_vlm_load.py --mock    # 랜덤 초기화 (다운로드 불필요)
"""
import sys, os, argparse, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.backbones.paligemma_backbone import PaliGemmaBackbone
from models.pi0_core import Pi0VLA

def main(mock: bool):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Device : {device}  |  dtype: {dtype}  |  mock: {mock}")

    # ── 1. 백본 로드 ──────────────────────────────────────────────
    t0 = time.time()
    backbone = PaliGemmaBackbone(
        load_pretrained=not mock,
        freeze_vision=True,
        freeze_language=False,
        dtype=dtype,
    ).to(device)
    print(f"Backbone loaded  ({time.time()-t0:.1f}s)  out_dim={backbone.out_dim}")

    # ── 2. Pi0VLA 조립 ────────────────────────────────────────────
    model = Pi0VLA(
        backbone=backbone,
        action_dim=3,
        horizon=10,
        hidden_dim=512,
        backbone_out_dim=backbone.out_dim,
    ).to(device)

    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Params total={total:.1f}M  trainable={trainable:.1f}M")

    # ── 3. 더미 입력으로 forward ──────────────────────────────────
    B = 1
    images = torch.randn(B, 8, 3, 224, 224, device=device, dtype=dtype)
    instructions = ["navigate to the charging station"]
    actions_gt   = torch.randn(B, 10, 3, device=device, dtype=dtype)

    t1 = time.time()
    loss = model.compute_loss(images, instructions, actions_gt.float())
    print(f"Loss : {loss.item():.4f}  ({time.time()-t1:.2f}s)")

    t2 = time.time()
    acts = model.sample_actions(images, instructions, n_steps=5, solver='heun')
    print(f"Actions shape : {acts.shape}  ({time.time()-t2:.2f}s)")

    assert acts.shape == (B, 10, 3), f"shape mismatch: {acts.shape}"
    print("✅ VLM 파이프라인 테스트 통과")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mock', action='store_true', help='랜덤 초기화 (가중치 불필요)')
    args = parser.parse_args()
    main(args.mock)
