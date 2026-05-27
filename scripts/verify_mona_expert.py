import torch
import sys
import os

# 프로젝트 루트 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.heads.mona_action_expert import MoNaActionExpert

def verify_numerical_alignment():
    print("=== MoNaActionExpert Numerical Alignment Verification ===")
    
    # 1. Instantiate Expert
    expert = MoNaActionExpert(action_dim=3, horizon=10)
    
    # 2. Test Normalization
    # MoNa v5 typical max action is around 1.15 for vx, vy
    raw_actions = torch.tensor([[[1.15, -1.15, 0.5]]]) # (B=1, T=1, D=3)
    normalized = expert.normalize(raw_actions)
    
    print(f"Raw Action: {raw_actions.tolist()}")
    print(f"Normalized: {normalized.tolist()}")
    
    # Check if normalized values are roughly around 1.0 (since dev=1.2)
    assert torch.all(torch.abs(normalized) < 1.1), "Normalization scale failure!"
    print("V Normalization: OK (Values scaled down to ~1.0 range)")

    # 3. Test Unnormalization (Recovery)
    # Simulate model output noise in range [-1, 1]
    noise_output = torch.tensor([[[0.9, -0.9, 0.5]]]) # 3rd dim increased to 0.5
    recovered = expert.unnormalize(noise_output)
    
    print(f"Model Noise Output: {noise_output.tolist()}")
    print(f"Recovered Physical Action: {recovered.tolist()}")
    
    # Check recovery scale (0.5 * 1.2 = 0.6)
    assert torch.all(torch.abs(recovered) > 0.5), "Unnormalization recovery failure!"
    print("V Unnormalization: OK (Values recovered to physical scale)")

    # 4. End-to-End Consistency
    cycle = expert.unnormalize(expert.normalize(raw_actions))
    diff = torch.abs(raw_actions - cycle).max()
    print(f"Max Reconstruction Error: {diff.item():.6f}")
    assert diff < 1e-5, "Numerical consistency failed!"
    print("V Consistency: OK")

    print("\n=== ALL NUMERICAL TESTS PASSED ===")

if __name__ == "__main__":
    verify_numerical_alignment()
