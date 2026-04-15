import torch
import h5py
import numpy as np
import os
import sys
from PIL import Image
from torchvision import transforms
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 프로젝트 루트 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pi0_core import Pi0VLA

def inspect_v5_dataset_output():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Dataset v5 파일 선택
    dataset_path = "/home/soda/MoNaVLA/ROS_action/mobile_vla_dataset_v5"
    h5_files = [f for f in os.listdir(dataset_path) if f.endswith('.h5')]
    sample_file = os.path.join(dataset_path, h5_files[0])
    print(f"Loading sample episode: {h5_files[0]}")

    with h5py.File(sample_file, 'r') as f:
        # observations/images: (N, 720, 1280, 3)
        images_all = f['observations/images'][:]
        # actions: (N, 3) -> [linear_x, linear_y, angular_z]
        actions_gt = f['actions'][:]
        # language_instruction
        instruction = f['language_instruction'][0].decode('utf-8') if isinstance(f['language_instruction'][0], bytes) else f['language_instruction'][0]

    print(f"Instruction: '{instruction}'")
    print(f"Total steps in episode: {len(images_all)}")

    # 2. 모델 로드 (FP16 최적화)
    print("Loading Pi0VLA model...")
    model = Pi0VLA(
        action_dim=3,
        horizon=10,
        vision_model_id="google/siglip-so400m-patch14-384",
        lang_model_id="google/gemma-2b"
    ).to(device, dtype=torch.float16)

    # 3. 입력 데이터 준비 (첫 번째 프레임)
    img_raw = images_all[0]
    img_pil = Image.fromarray(img_raw)
    
    # SigLIP 입력 규격에 맞게 리사이즈 및 텐서 변환
    preprocess = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensor = preprocess(img_pil).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float16) # (B, Window, C, H, W)
    instructions = [instruction]

    # 4. 모델 실행 (Action Sampling)
    print("\nGenerating actions via Flow Matching (untrained model)...")
    with torch.no_grad():
        # 생성할 미래 액션 스텝 수는 10 (Horizon)
        sampled_actions = model.sample_actions(img_tensor, instructions, n_steps=5)
    
    # [1, 10, 3] -> (10, 3)
    generated = sampled_actions[0].cpu().numpy()

    # 5. 결과 출력 (Ground Truth vs Generated)
    print("\n" + "="*60)
    print(f"{'Step':^6} | {'GT Action (vx, vy, wz)':^25} | {'Generated (untrained)':^25}")
    print("-" * 60)
    
    # GT는 에피소드 전체 기록 중 앞부분 10개 출력
    for i in range(10):
        gt_str = f"{actions_gt[i][0]:.3f}, {actions_gt[i][1]:.3f}, {actions_gt[i][2]:.3f}" if i < len(actions_gt) else "N/A"
        gen_str = f"{generated[i][0]:.3f}, {generated[i][1]:.3f}, {generated[i][2]:.3f}"
        print(f"{i+1:^6} | {gt_str:^25} | {gen_str:^25}")
    print("="*60)
    print("\n* GT Action: 실제 로봇 전송/기록된 값 (vx, vy=0, wz)")
    print("* Generated: untrained π0 모델이 Flow Matching으로 생성한 가우시안 노이즈 기반 액션")

if __name__ == "__main__":
    inspect_v5_dataset_output()
