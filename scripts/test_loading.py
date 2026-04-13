import torch
import torch.nn as nn
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
import os
import sys

# 프로젝트 루트 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pi0_core import Pi0VLA

def test_model_loading():
    print("Checking device...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "microsoft/kosmos-2-patch14-224"
    print(f"Loading backbone: {model_name}...")
    
    try:
        # 1. Load Backbone (Kosmos-2)
        processor = AutoProcessor.from_pretrained(model_name)
        backbone = Kosmos2ForConditionalGeneration.from_pretrained(model_name).to(device)
        print("Backbone loaded successfully.")
        
        # 2. Instantiate MoNa-pi
        print("Instantiating Pi0VLA...")
        model = Pi0VLA(
            backbone=backbone,
            action_dim=3,
            horizon=10,
            backbone_out_dim=768 # Kosmos-2 hidden size (patch14-224 version)
        ).to(device)
        print("Pi0VLA instantiated successfully.")
        
        # 3. Simple Forward Pass (Backbone Placeholder Test)
        dummy_images = torch.randn(1, 1, 3, 224, 224).to(device)
        dummy_instructions = ["move forward"]
        
        print("Testing forward pass (sampling)...")
        # NOTE: pi0_core.py의 forward_backbone은 현재 Placeholder입니다.
        actions = model.sample_actions(dummy_images, dummy_instructions, n_steps=5)
        print(f"Sampled actions shape: {actions.shape}")
        print("Loading test PASSED.")
        
    except Exception as e:
        print(f"Loading test FAILED: {e}")

if __name__ == "__main__":
    test_model_loading()
