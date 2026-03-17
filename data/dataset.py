import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image as PILImage
from pathlib import Path

class ActionNormalizer:
    """
    Action Normalization for 3-Omniwheel Robot (lx, ly, az)
    Based on mobile_vla_dataset_v3 analysis:
    - linear_x: -1.15 ~ 1.15 
    - linear_y: -1.15 ~ 1.15 (lateral)
    - angular_z: -1.15 ~ 1.15 (rotation)
    """
    def __init__(self):
        # 99% 확실한 통계치 (1.15 max)
        self.min_vals = torch.tensor([-1.15, -1.15, -1.15])
        self.max_vals = torch.tensor([1.15, 1.15, 1.15])
        
    def normalize(self, action):
        """Map [min, max] to [-1, 1]"""
        return 2.0 * (action - self.min_vals) / (self.max_vals - self.min_vals + 1e-8) - 1.0
    
    def unnormalize(self, action_norm):
        """Map [-1, 1] to [min, max]"""
        return 0.5 * (action_norm + 1.0) * (self.max_vals - self.min_vals) + self.min_vals

class ActionChunkDataset(Dataset):
    """
    HDF5 Dataset with Action Chunking for MoNa-pi
    Standard: Window size 8, Prediction Horizon 10 (Total 18 frames)
    """
    def __init__(self, directory, k=10, window_size=8, transform=None):
        self.directory = Path(directory)
        self.h5_files = sorted(list(self.directory.glob("*.h5")))
        self.k = k  # Horizon
        self.window_size = window_size
        self.transform = transform
        self.normalizer = ActionNormalizer()
        
        # Pre-index samples
        self.samples = []
        for f_idx, f_path in enumerate(self.h5_files):
            try:
                with h5py.File(f_path, 'r') as f:
                    n_frames = f['images'].shape[0]
                    # window_size개의 이미지를 보고 k개의 액션을 예측할 수 있는 시작점들 찾기
                    for t in range(self.window_size - 1, n_frames - self.k + 1):
                        self.samples.append((f_idx, t))
            except Exception as e:
                print(f"Skipping {f_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f_idx, t = self.samples[idx]
        f_path = self.h5_files[f_idx]
        
        with h5py.File(f_path, 'r') as f:
            # 1. Images (History Window of size self.window_size)
            # t sits at the end of the window
            imgs = f['images'][t - self.window_size + 1 : t + 1] 
            
            # 2. Actions (Future Chunk of size self.k, starting at t)
            actions = f['actions'][t : t + self.k] # (k, 3)
            
            # 3. Instruction
            instr = f.attrs.get('task', "Navigate to the goal")
            if isinstance(instr, bytes):
                instr = instr.decode('utf-8')
            
        # Image Processing
        processed_images = []
        for img in imgs:
            img_pil = PILImage.fromarray(img)
            if self.transform:
                img_pil = self.transform(img_pil)
            # Standard normalization to [0, 1]
            processed_images.append(torch.from_numpy(np.array(img_pil).transpose(2, 0, 1)).float() / 255.0)
            
        images = torch.stack(processed_images) # (W, C, H, W)
        actions = torch.from_numpy(actions).float()
        actions = self.normalizer.normalize(actions)
        
        return {
            "images": images,
            "actions": actions,
            "instructions": instr
        }
