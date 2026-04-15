import torch
import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pi0_core import Pi0VLA

def test_loss_and_sampling():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 모델 초기화
    model = Pi0VLA(
        action_dim=3,
        horizon=10,
        hidden_dim=512,
    ).to(device)
    
    batch_size = 2
    # SigLIP SO400M expects 384x384; model runs in fp16 (self.half() in __init__)
    images = torch.randn(batch_size, 8, 3, 384, 384).to(device).half()
    instructions = ["move forward", "turn left"]
    # Chunk size 10, Action dim 3
    actions_gt = torch.randn(batch_size, 10, 3).to(device).half()

    # 1. Loss 계산 테스트
    loss = model.compute_loss(images, instructions, actions_gt)
    print(f"Loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"

    # 2. 샘플링 테스트 (Inference)
    sampled_actions = model.sample_actions(images, instructions, n_steps=5)
    print(f"Sampleed actions shape: {sampled_actions.shape}")
    assert sampled_actions.shape == (batch_size, 10, 3), f"Wrong shape: {sampled_actions.shape}"
    
    print("✅ Logic Verification Success!")

if __name__ == "__main__":
    test_loss_and_sampling()
