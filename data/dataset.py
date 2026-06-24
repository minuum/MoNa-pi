import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image as PILImage
from pathlib import Path

from data.preprocessing import EpisodePreprocessor, ActionNormalizer, CLIP_MEAN, CLIP_STD
from data.ood_augment import sample_ood_params, apply_ood_aug


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


def _read_instruction(h5file: h5py.File, fmt: str) -> str:
    """
    v5: language_instruction 은 dataset (shape (1,), dtype object)
    v3: task 는 attrs
    """
    if fmt == "v5":
        if "language_instruction" in h5file:
            raw = h5file["language_instruction"][0]
        else:
            raw = h5file.attrs.get("language_instruction", "Navigate to the goal")
    else:
        raw = h5file.attrs.get("task", "Navigate to the goal")

    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def _read_episode(h5file: h5py.File, fmt: str):
    """포맷에 맞게 images, actions, instruction 읽기"""
    if fmt == "v5":
        images = h5file["observations/images"][:]   # (T, H, W, 3)
        actions = h5file["actions"][:]              # (T, 3)
    else:  # v3
        images = h5file["images"][:]
        actions = h5file["actions"][:]

    instr = _read_instruction(h5file, fmt)
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
        normalize: bool = False,   # Pi0 정통 방식: 모델 내부에서 정규화하므로 기본값 False
        inject_prefix: bool = True,
        use_delta: bool = False,
        augment: bool = False,
        use_color_jitter: bool = False,
        use_random_crop: bool = False,
        use_counterfactual: bool = False,
        use_ood_aug: bool = False,
        ood_aug_p: float = 0.35,
        is_training: bool = True,
        file_list: list = None,
    ):
        self.directory = Path(directory)
        self.h5_files = file_list if file_list is not None else sorted(self.directory.glob("*.h5"))
        if not self.h5_files:
            raise FileNotFoundError(f"H5 파일을 찾을 수 없음: {self.directory}")

        self.k = k
        self.window_size = window_size
        self.image_size = image_size
        self.transform = transform
        self.augment = augment
        self.use_ood_aug = use_ood_aug
        self.ood_aug_p = ood_aug_p
        self.is_training = is_training

        # 전처리 파이프라인 초기화
        self.preprocessor = EpisodePreprocessor(
            smooth=preprocess and smooth,
            normalize=normalize,
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

        # OOD 흉내 증강 파라미터 — 윈도우 전체에 동일하게 적용(프레임별 독립 샘플링 X)
        ood_params = (
            sample_ood_params(self.ood_aug_p)
            if (self.use_ood_aug and self.is_training)
            else {}
        )

        processed_images = []
        for img in imgs_raw:
            img_pil = PILImage.fromarray(img).resize(
                (self.image_size, self.image_size), PILImage.BILINEAR
            )
            # OOD 흉내 증강 — robot_close/far(zoom), basket_left/right_extreme(shift)
            if ood_params:
                img_pil = apply_ood_aug(img_pil, ood_params)
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


def get_free_holdout_files(directory: str) -> list:
    """`free_*` 에피소드 전체 (train/val 분할 대상에서 제외되는 고정 OOD holdout)."""
    return sorted(f for f in Path(directory).glob("*.h5") if "free_" in f.name)


def build_free_holdout(
    directory: str,
    k: int = 10,
    window_size: int = 8,
    image_size: int = 384,
    preprocess: bool = True,
    normalize: bool = False,
) -> "ActionChunkDataset":
    """
    `free_*` 에피소드 전체로 구성된 고정 평가용 데이터셋.

    train/val 분할에서 완전히 빠지므로 매 실험마다 동일한 모집단으로
    OOD(free_*) SR을 비교할 수 있다. 증강/counterfactual 없음(평가 전용).
    """
    free_files = get_free_holdout_files(directory)
    if not free_files:
        raise FileNotFoundError(f"free_* H5 파일을 찾을 수 없음: {directory}")
    return ActionChunkDataset(
        directory=directory,
        file_list=free_files,
        k=k,
        window_size=window_size,
        image_size=image_size,
        preprocess=preprocess,
        normalize=normalize,
        is_training=False,
    )


def build_train_val_split(
    directory: str,
    val_split: float = 0.1,
    k: int = 10,
    window_size: int = 8,
    image_size: int = 384,
    preprocess: bool = True,
    normalize: bool = False,
    augment: bool = False,
    use_color_jitter: bool = False,
    use_random_crop: bool = False,
    use_counterfactual: bool = False,
    use_ood_aug: bool = False,
    ood_aug_p: float = 0.35,
    seed: int = 42,
    exclude_free_holdout: bool = True,
):
    """
    데이터셋을 train/val 로 나눠 반환.
    train: augment/counterfactual 활성화 / val: 비활성화

    에피소드(h5 파일) 단위로 먼저 분할한 뒤 각자 윈도우를 생성한다.
    (이전엔 윈도우 생성 후 윈도우 단위로 random_split — window_size=8,
    k=10 윈도우끼리는 프레임 7/8이 겹쳐서 같은 에피소드의 인접 윈도우가
    train/val에 양쪽으로 갈리는 누출이 있었음. 92.5%의 "val 에피소드"가
    실제로는 train에도 자기 자신의 다른 윈도우를 갖고 있었던 것으로 확인됨,
    2026-06-24 — MoNaVLA의 robovlm_nav/datasets/nav_h5_dataset_impl.py처럼
    파일 단위로 먼저 나눠 이 문제를 제거.)

    exclude_free_holdout=True(기본, 2026-06-24~): `free_*` 에피소드는 이
    train/val 분할에 전혀 포함되지 않는다. 대신 `build_free_holdout()`이
    반환하는 고정 데이터셋으로 따로 평가한다. 이전엔 stratify_free로
    val에 일부(seed당 1~2개)만 끼워 넣었는데, 실험마다 어떤 free_*가
    val에 들어가는지가 흔들려서 SR 비교가 불안정했음(n=2 수준).
    `free_*`를 학습에 추가해도 SR이 안 바뀐다는 게 이미 M7/M8에서
    확인됐으므로, 학습 신호로서의 손실보다 고정 모집단으로 평가
    안정성을 얻는 게 더 가치 있다고 판단함.

    Returns:
        train_dataset, val_dataset
    """
    all_files = sorted(Path(directory).glob("*.h5"))
    if not all_files:
        raise FileNotFoundError(f"H5 파일을 찾을 수 없음: {directory}")

    def _split_group(files, ratio):
        shuffled = list(files)
        random.Random(seed).shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * ratio)) if files else 0
        return shuffled[:n_val], shuffled[n_val:]

    if exclude_free_holdout:
        regular_files = [f for f in all_files if "free_" not in f.name]
        val_files, train_files = _split_group(regular_files, val_split)
        val_files, train_files = sorted(val_files), sorted(train_files)
    else:
        shuffled = all_files.copy()
        random.Random(seed).shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_split))
        val_files = sorted(shuffled[:n_val])
        train_files = sorted(shuffled[n_val:])

    train_ds = ActionChunkDataset(
        directory=directory,
        file_list=train_files,
        k=k,
        window_size=window_size,
        image_size=image_size,
        preprocess=preprocess,
        normalize=normalize,
        augment=augment,
        use_color_jitter=use_color_jitter,
        use_random_crop=use_random_crop,
        use_counterfactual=use_counterfactual,
        use_ood_aug=use_ood_aug,
        ood_aug_p=ood_aug_p,
        is_training=True,
    )
    val_ds = ActionChunkDataset(
        directory=directory,
        file_list=val_files,
        k=k,
        window_size=window_size,
        image_size=image_size,
        preprocess=preprocess,
        normalize=normalize,
        augment=False,
        use_color_jitter=False,
        use_random_crop=False,
        use_counterfactual=False,
        is_training=False,
    )

    return train_ds, val_ds
