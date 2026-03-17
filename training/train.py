import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import os
import sys

# 프로젝트 루트 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pi0_core import Pi0VLA
from data.dataset import ActionChunkDataset

def load_pretrained_weights(model, ckpt_path):
    """
    기존 VLM 백본 가중치 로드
    """
    if not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint not found at {ckpt_path}. Starting from scratch.")
        return model
        
    print(f"Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # 모델의 state_dict와 체크포인트의 키 매칭 (백본 위주)
    # Flow Head는 구조가 다르므로 무시하거나 부분 매칭 필요
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"Successfully loaded {len(pretrained_dict)} layers.")
    return model

def train():
    accelerator = Accelerator()
    device = accelerator.device
    
    # 1. Config & Hyperparameters
    k = 10 # Chunk size (99% certainty)
    window_size = 8 # Window size (99% certainty)
    batch_size = 4 # Reduced for memory safety with windowed images
    lr = 1e-4
    epochs = 10
    data_dir = "/home/soda/vla/ROS_action/mobile_vla_dataset_v3/"
    ckpt_path = "/home/soda/vla/epoch_epoch=09-val_loss=val_loss=0.010.ckpt" 
    
    # 2. Dataset & Dataloader
    dataset = ActionChunkDataset(directory=data_dir, k=k, window_size=window_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 3. Model Setup (Placeholder backbone)
    # 실제 환경에서는 HF에서 로드하거나 기존 가중치를 불러옴
    backbone = nn.Identity() # Placeholder
    model = Pi0VLA(
        backbone=backbone,
        action_dim=3,
        horizon=k,
        backbone_out_dim=1024 
    )
    
    # Pre-trained weights 로드
    model = load_pretrained_weights(model, ckpt_path)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # 4. Prepare for Distributed Training
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )
    
    # 5. Training Loop
    model.train()
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            images = batch['images']
            actions = batch['actions']
            instructions = batch['instructions']
            
            optimizer.zero_grad()
            
            loss = model.compute_loss(images, instructions, actions)
            
            accelerator.backward(loss)
            optimizer.step()
            
            progress_bar.set_postfix(loss=loss.item())
            
        # 6. Save Checkpoint
        if accelerator.is_local_main_process:
            os.makedirs("checkpoints", exist_ok=True)
            accelerator.save_model(model, f"checkpoints/mona_pi_epoch_{epoch}")
            
    print("Training Completed.")

if __name__ == "__main__":
    train()
