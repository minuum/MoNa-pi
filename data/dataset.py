import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, random_split
import h5py
import numpy as np
from PIL import Image as PILImage
from pathlib import Path

from data.preprocessing import EpisodePreprocessor, ActionNormalizer, CLIP_MEAN, CLIP_STD


# ─────────────────────────────────────────────────────
# 하위 호환용 ActionNormalizer (기존 train.py 참조용)
# ─────────────────────────────────────────────────────
# (preprocessing.py 의 ActionNormalizer 를 직접 임포트해서 써도 됨)


def _detect_format(h5file: h5py.File) -> str:
    """
    HDF5 파일의 데이터 포맷 버전을 자동 감지.

    - v5/v6: 'observations/images', 'actions', 'language_instruction' 속성
    - v3:    'images', 'actions', 'task' 속성
    """
    if "observations/images" in h5file:
        return "v5"
    if "images" in h5file:
        return "v3"
    raise ValueError(f"지원하지 않는 HDF5 구조: {list(h5file.keys())}")


def _read_episode(h5file: h5py.File, fmt: str):
    """포맷에 맞게 images, actions, instruction 읽기"""
    if fmt == "v5":
        images = h5file["observations/images"][:]   # (T, H, W, 3)
        actions = h5file["actions"][:]              # (T, 3)
        instr = h5file.attrs.get("language_instruction", "Navigate to the goal")
    else:  # v3
        images = h5file["images"][:]
        actions = h5file["actions"][:]
        instr = h5file.attrs.get("task", "Navigate to the goal")

    if isinstance(instr, bytes):
        instr = instr.decode("utf-8")
    return images, actions.astype(np.float32), instr


class ActionChunkDataset(Dataset):
    """
    HDF5 Dataset with Action Chunking for MoNa-pi

    MoNaVLA 계승:
    - CLIP 이미지 정규화 (mean/std)
    - HFlip 증강 + angular_z 부호 반전 + instruction left↔right 교체
    - ColorJitter / RandomCrop 증강
    - CounterfactualInjector (stop/steer 오버라이드)

    Args:
        directory:          HDF5 파일들이 있는 디렉토리 경로
        k:                  예측 Horizon (기본 10)
        window_size:        이미지 이력 윈도우 (기본 8)
        image_size:         리사이즈 해상도 (기본 384, SigLIP 기준)
        preprocess:         ActionSmoother + IntentPrefix 활성화
        smooth:             Savitzky-Golay 스무딩
        inject_prefix:      9-class 의도 Prefix 주입
        use_delta:          속도→변위 변환 (실험용)
        augment:            HFlip 증강 활성화
        use_color_jitter:   ColorJitter 증강
        use_random_crop:    RandomResizedCrop 증강
        use_counterfactual: Counterfactual 학습 (학습 시만 유효)
        is_training:        학습/검증 구분 (counterfactual, noise 비활성화 여부)
    """

    def __init__(
        self,
        directory: str,
        k: int = 10,
        window_size: int = 8,
        image_size: int = 384,
        transform=None,
        preprocess: bool = True,
        smooth: bool = True,
        inject_prefix: bool = True,
        use_delta: bool = False,
        augment: bool = False,
        use_color_jitter: bool = False,
        use_random_crop: bool = False,
        use_counterfactual: bool = False,
        is_training: bool = True,
    ):
        self.directory = Path(directory)
        self.h5_files = sorted(self.directory.glob("*.h5"))
        if not self.h5_files:
            raise FileNotFoundError(f"H5 파일을 찾을 수 없음: {self.directory}")

        self.k = k
        self.window_size = window_size
        self.image_size = image_size
        self.transform = transform
        self.augment = augment
        self.is_training = is_training

        # 전처리 파이프라인 초기화
        self.preprocessor = EpisodePreprocessor(
            smooth=preprocess and smooth,
            normalize=True,
            inject_prefix=preprocess and inject_prefix,
            use_delta=preprocess and use_delta,
            use_counterfactual=use_counterfactual,
        )

        # MoNaVLA 계승: CLIP 정규화 텐서
        self._clip_mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        self._clip_std  = torch.tensor(CLIP_STD).view(3, 1, 1)

        # 증강 모듈 (MoNaVLA 계승)
        self._color_jitter = T.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        ) if use_color_jitter else None
        self._random_crop = T.RandomResizedCrop(
            image_size, scale=(0.8, 1.0)
        ) if use_random_crop else None

        # ── 에피소드를 메모리에 캐시 (actions만 전처리 후 저장) ─────────
        # images는 용량이 크므로 파일 경로만 인덱싱
        self.samples = []      # (f_idx, t) 리스트
        self._ep_cache = {}    # f_idx → (images, processed_actions, instruction, fmt)

        for f_idx, f_path in enumerate(self.h5_files):
            try:
                with h5py.File(f_path, "r") as f:
                    fmt = _detect_format(f)
                    images, raw_actions, instr = _read_episode(f, fmt)

                # 에피소드 단위 액션 전처리 (스무딩 등)
                proc_actions = self.preprocessor.process_episode_actions(raw_actions)

                self._ep_cache[f_idx] = {
                    "images": images,
                    "actions": proc_actions,
                    "instruction": instr,
                }

                n_frames = images.shape[0]
                for t in range(window_size - 1, n_frames - k + 1):
                    self.samples.append((f_idx, t))

            except Exception as e:
                print(f"[Dataset] 건너뜀 {f_path.name}: {e}")

        print(f"[Dataset] {len(self.h5_files)}개 에피소드, {len(self.samples)}개 샘플 로드 완료")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f_idx, t = self.samples[idx]
        ep = self._ep_cache[f_idx]

        images_raw = ep["images"]
        proc_actions = ep["actions"]
        base_instr = ep["instruction"]

        # ── 이미지 윈도우 ─────────────────────────────────────────
        imgs_raw = images_raw[t - self.window_size + 1 : t + 1]

        processed_images = []
        for img in imgs_raw:
            img_pil = PILImage.fromarray(img).resize(
                (self.image_size, self.image_size), PILImage.BILINEAR
            )
            # ColorJitter (MoNaVLA 계승)
            if self._color_jitter is not None:
                img_pil = self._color_jitter(img_pil)
            # RandomCrop (MoNaVLA 계승)
            if self._random_crop is not None:
                img_pil = self._random_crop(img_pil)
            if self.transform:
                img_pil = self.transform(img_pil)
            arr = np.array(img_pil).transpose(2, 0, 1).astype(np.float32) / 255.0
            t_img = torch.from_numpy(arr)
            # CLIP 정규화 (MoNaVLA 계승 — SigLIP도 동일 상수 사용)
            t_img = (t_img - self._clip_mean) / self._clip_std
            processed_images.append(t_img)

        images = torch.stack(processed_images)  # (window_size, C, H, W)

        # ── 액션 청크 ────────────────────────────────────────────
        action_chunk = proc_actions[t : t + self.k]  # (k, 3)

        # ── Counterfactual 주입 (학습 시만) ──────────────────────
        action_chunk, base_instr = self.preprocessor.apply_counterfactual(
            action_chunk, base_instr, is_training=self.is_training
        )

        # ── HFlip 증강 (MoNaVLA 계승) ─────────────────────────────
        if self.augment and self.is_training and random.random() < 0.5:
            images = torch.stack([TF.hflip(img) for img in images])
            # angular_z(dim=2) 부호 반전
            action_chunk = action_chunk.copy()
            action_chunk[:, 2] = -action_chunk[:, 2]
            base_instr = self.preprocessor.flip_instruction(base_instr)

        # ── 정규화 ────────────────────────────────────────────────
        action_chunk_norm = self.preprocessor.normalize_chunk(action_chunk)
        actions = torch.from_numpy(action_chunk_norm).float()

        # ── 의도 Prefix + 다국어 instruction 생성 ────────────────
        instruction = self.preprocessor.get_instruction(
            base_instr, action_chunk, is_training=self.is_training
        )

        return {
            "images": images,           # (window_size, C, H, W)
            "actions": actions,         # (k, 3)  정규화됨
            "instructions": instruction,
        }

    @property
    def normalizer(self):
        """하위 호환 접근자 — train.py 등에서 normalizer.unnormalize() 호출 시 사용"""
        return self.preprocessor.normalizer


def build_train_val_split(
    directory: str,
    val_split: float = 0.1,
    k: int = 10,
    window_size: int = 8,
    image_size: int = 384,
    preprocess: bool = True,
    augment: bool = False,
    use_color_jitter: bool = False,
    use_random_crop: bool = False,
    use_counterfactual: bool = False,
    seed: int = 42,
):
    """
    데이터셋을 train/val 로 나눠 반환.
    train: augment/counterfactual 활성화 / val: 비활성화

    Returns:
        train_dataset, val_dataset
    """
    train_full = ActionChunkDataset(
        directory=directory,
        k=k,
        window_size=window_size,
        image_size=image_size,
        preprocess=preprocess,
        augment=augment,
        use_color_jitter=use_color_jitter,
        use_random_crop=use_random_crop,
        use_counterfactual=use_counterfactual,
        is_training=True,
    )
    val_full = ActionChunkDataset(
        directory=directory,
        k=k,
        window_size=window_size,
        image_size=image_size,
        preprocess=preprocess,
        augment=False,
        use_color_jitter=False,
        use_random_crop=False,
        use_counterfactual=False,
        is_training=False,
    )

    # 동일 seed로 같은 에피소드를 train/val로 분할
    n_val_t = max(1, int(len(train_full) * val_split))
    n_train = len(train_full) - n_val_t
    n_val_v = max(1, int(len(val_full) * val_split))

    generator = torch.Generator().manual_seed(seed)
    train_ds, _ = random_split(train_full, [n_train, n_val_t], generator=generator)
    generator2  = torch.Generator().manual_seed(seed)
    _, val_ds   = random_split(val_full,   [len(val_full) - n_val_v, n_val_v], generator=generator2)

    return train_ds, val_ds
