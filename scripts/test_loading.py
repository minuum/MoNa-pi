import torch
import torch.nn as nn
from transformers import AutoProcessor
import os
import sys
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 프로젝트 루트 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pi0_core import Pi0VLA

def test_original_pi0_loading():
    print("Checking device...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # 1. Instantiate MoNa-pi (Original Spec)
        # 생성자 내부에서 SigLIP(SO400M) 및 Gemma-2B를 로드합니다.
        print("Instantiating Pi0VLA (SigLIP-SO400M + Gemma-2B)...")
        
        # 메모리 절약을 위해 float16을 명시적으로 사용합니다.
        model = Pi0VLA(
            action_dim=3,
            horizon=10,
            hidden_dim=512,
            vision_model_id="google/siglip-so400m-patch14-384",
            lang_model_id="google/gemma-2b"
        ).to(device, dtype=torch.float16)
        
        print("Model instantiated successfully (FP16).")
        
        # 2. Simple Forward Pass Test
        # images: (B, Window, C, H, W) -> Pi0 Spec: H=384, W=384
        dummy_images = torch.randn(1, 1, 3, 384, 384).to(device, dtype=torch.float16)
        dummy_instructions = ["move forward and avoid the brown pot"]
        
        print("Testing forward pass (sampling)...")
        # Heun's method solver (5 steps)
        actions = model.sample_actions(dummy_images, dummy_instructions, n_steps=5)
        
        print(f"Sampled actions shape: {actions.shape}")
        print("=== TEST PASSED ===")
        
    except Exception as e:
        print(f"=== TEST FAILED ===")
        print(f"Error: {e}")
        
        # Debug: Check parameter/buffer dtypes
        print("\n[Debug] Non-Half Parameters/Buffers:")
        found_problem = False
        for name, param in model.named_parameters():
            if param.dtype != torch.float16:
                print(f" - Param {name}: {param.dtype} | {param.shape}")
                found_problem = True
        for name, buf in model.named_buffers():
            if buf.dtype != torch.float16:
                print(f" - Buffer {name}: {buf.dtype} | {buf.shape}")
                found_problem = True
        
        if not found_problem:
            print(" - All parameters and buffers are in half precision.")

        if "gated model" in str(e):
            print("\n[!] Error: Gemma-2B 접근 권한이 없습니다.")
            print("1. https://huggingface.co/google/gemma-2b 에서 이용 약관에 동의하세요.")
            print("2. .env 파일에 올바른 HF_TOKEN을 입력하세요.")

if __name__ == "__main__":
    test_original_pi0_loading()
