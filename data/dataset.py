import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import h5py
import numpy as np
from pathlib import Path


class ActionNormalizer:
    """
    3-DOF omniwheel: (linear_x, linear_y, angular_z), range ±1.15
    (stats from mobile_vla_dataset_v3/v5 analysis)
    """
    def __init__(self):
        self.min_vals = torch.tensor([-1.15, -1.15, -1.15])
        self.max_vals = torch.tensor([ 1.15,  1.15,  1.15])

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        return 2.0 * (action - self.min_vals) / (self.max_vals - self.min_vals + 1e-8) - 1.0

    def unnormalize(self, action_norm: torch.Tensor) -> torch.Tensor:
        return 0.5 * (action_norm + 1.0) * (self.max_vals - self.min_vals) + self.min_vals


def _detect_version(f: h5py.File) -> str:
    """HDF5 파일 버전 자동 감지 (v3: 'images' 직접, v5: 'observations/images')"""
    if 'observations' in f:
        return 'v5'
    return 'v3'


class ActionChunkDataset(Dataset):
    """
    HDF5 Dataset — v3 / v5 자동 감지.

    v3 구조: images (N,H,W,3), actions (N,3), attrs['task']
    v5 구조: observations/images (N,H,W,3), actions (N,3), language_instruction (1,)
    """

    IMG_SIZE = 224  # SigLIP input size

    def __init__(
        self,
        directory: str,
        k: int = 10,
        window_size: int = 8,
        augment: bool = False,
    ):
        self.directory = Path(directory)
        self.h5_files = sorted(self.directory.glob("*.h5"))
        if not self.h5_files:
            raise FileNotFoundError(f"No .h5 files found in {directory}")

        self.k = k
        self.window_size = window_size
        self.normalizer = ActionNormalizer()

        self.transform = T.Compose([
            T.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1) if augment else T.Lambda(lambda x: x),
            T.ToTensor(),           # → (C, H, W), [0, 1]
        ])

        # 유효 샘플 인덱싱
        self.samples: list[tuple[int, int]] = []
        for f_idx, f_path in enumerate(self.h5_files):
            try:
                with h5py.File(f_path, 'r') as f:
                    ver = _detect_version(f)
                    imgs_ds = f['observations']['images'] if ver == 'v5' else f['images']
                    n = imgs_ds.shape[0]
                for t in range(window_size - 1, n - k + 1):
                    self.samples.append((f_idx, t))
            except Exception as e:
                print(f"[dataset] skip {f_path.name}: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        f_idx, t = self.samples[idx]
        f_path = self.h5_files[f_idx]

        with h5py.File(f_path, 'r') as f:
            ver = _detect_version(f)

            if ver == 'v5':
                raw_imgs = f['observations']['images'][t - self.window_size + 1 : t + 1]
                instr_raw = f['language_instruction'][0]
            else:
                raw_imgs = f['images'][t - self.window_size + 1 : t + 1]
                instr_raw = f.attrs.get('task', "Navigate to the goal")

            actions = f['actions'][t : t + self.k]   # (k, 3)

        # instruction decode
        if isinstance(instr_raw, (bytes, np.bytes_)):
            instr = instr_raw.decode('utf-8')
        else:
            instr = str(instr_raw)

        # image processing
        from PIL import Image as PILImage
        imgs = []
        for img_np in raw_imgs:
            pil = PILImage.fromarray(img_np.astype(np.uint8))
            imgs.append(self.transform(pil))
        images = torch.stack(imgs)   # (W, 3, 224, 224)

        actions = torch.from_numpy(np.array(actions)).float()
        actions = self.normalizer.normalize(actions)

        return {"images": images, "actions": actions, "instructions": instr}
