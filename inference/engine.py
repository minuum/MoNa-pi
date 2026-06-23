"""
MoNa-pi 추론 엔진

특징:
    - 비전 특징 캐싱: 같은 이미지 프레임 반복 요청 시 백본 재계산 생략
    - BF16 일관 유지 (FP16은 NaN 유발)
    - 교체 가능한 ODE 솔버 (euler / heun / dpm)
    - 전처리(정규화) + 후처리(역정규화) 내장

사용:
    engine = MoNaPiEngine("checkpoints/best", device="cuda", solver="heun", n_ode_steps=5)
    engine.warmup()
    actions = engine.predict(image_np, instruction="장애물 회피")
    # actions: np.ndarray (10, 3)  [vx, vy, wz] in m/s
"""

from __future__ import annotations

import gc
import hashlib
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pi0_core import Pi0VLA
from inference.ode_solver import build_solver


class ActionNormalizer:
    """3-DOF 옴니휠 로봇 액션 정규화 (lx, ly, az). 범위: [-1.15, 1.15]"""

    def __init__(self, min_val: float = -1.15, max_val: float = 1.15, action_dim: int = 3):
        self.min_vals = np.full(action_dim, min_val, dtype=np.float32)
        self.max_vals = np.full(action_dim, max_val, dtype=np.float32)

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        return 2.0 * (actions - self.min_vals) / (self.max_vals - self.min_vals + 1e-8) - 1.0

    def unnormalize(self, actions_norm: np.ndarray) -> np.ndarray:
        return 0.5 * (actions_norm + 1.0) * (self.max_vals - self.min_vals) + self.min_vals

    def normalize_tensor(self, actions: torch.Tensor) -> torch.Tensor:
        min_t = torch.from_numpy(self.min_vals).to(actions.device)
        max_t = torch.from_numpy(self.max_vals).to(actions.device)
        return 2.0 * (actions - min_t) / (max_t - min_t + 1e-8) - 1.0

    def unnormalize_tensor(self, actions: torch.Tensor) -> torch.Tensor:
        min_t = torch.from_numpy(self.min_vals).to(actions.device)
        max_t = torch.from_numpy(self.max_vals).to(actions.device)
        return 0.5 * (actions + 1.0) * (max_t - min_t) + min_t


class MoNaPiEngine:
    """
    MoNa-pi 모델 추론 엔진.

    Args:
        model_path:                  체크포인트 디렉토리 경로
        device:                      "cuda" or "cpu"
        solver:                      "euler" | "heun" | "dpm"
        n_ode_steps:                 ODE 적분 스텝 수
        action_dim:                  로봇 액션 차원 (기본 3: vx, vy, wz)
        horizon:                     예측 청크 크기 (기본 10)
        hidden_dim:                  Action Expert 히든 차원
        use_paligemma:               True: PaliGemmaBackbone (π0 방식)
        load_pretrained_paligemma:   True: HF에서 paligemma-3b 다운로드
        vision_model_id:             로컬 SigLIP 모델 ID
        lang_model_id:               로컬 Gemma 모델 ID
        paligemma_id:                PaliGemma HF 모델 ID
        cache_vision:                동일 이미지 연속 요청 시 VLM 캐시 여부
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        solver: str = "heun",
        n_ode_steps: int = 5,
        action_dim: int = 3,
        horizon: int = 10,
        hidden_dim: int = 512,
        use_paligemma: bool = True,
        load_pretrained_paligemma: bool = False,
        use_int8: bool = False,
        vision_model_id: str = "google/siglip-so400m-patch14-384",
        lang_model_id: str = "google/gemma-2b",
        paligemma_id: str = "google/paligemma-3b-pt-224",
        image_size: int = 224,
        cache_vision: bool = True,
    ):
        self.model_path = Path(model_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.solver = build_solver(solver, n_ode_steps)
        self.n_ode_steps = n_ode_steps
        self.action_dim = action_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.use_paligemma = use_paligemma
        self.load_pretrained_paligemma = load_pretrained_paligemma
        self.use_int8 = use_int8
        self.vision_model_id = vision_model_id
        self.lang_model_id = lang_model_id
        self.paligemma_id = paligemma_id
        self.cache_vision = cache_vision
        self.IMG_SIZE = image_size

        self.model: Optional[Pi0VLA] = None
        self.normalizer = ActionNormalizer()

        # 비전 특징 캐시: {image_hash: (cond_tensor, timestamp)}
        self._vision_cache: dict[str, tuple[torch.Tensor, float]] = {}
        self._cache_ttl = 0.5  # 초: 이 시간이 지나면 캐시 무효화

    # ──────────────────────────────────────────
    # 초기화
    # ──────────────────────────────────────────

    def warmup(self):
        """모델 로드 + 더미 추론으로 CUDA 커널 예열"""
        print(f"[Engine] 모델 로드 중: {self.model_path}")
        self.model = Pi0VLA(
            action_dim=self.action_dim,
            horizon=self.horizon,
            hidden_dim=self.hidden_dim,
            use_paligemma=self.use_paligemma,
            load_pretrained_paligemma=self.load_pretrained_paligemma,
            use_int8=self.use_int8,
            vision_model_id=self.vision_model_id,
            lang_model_id=self.lang_model_id,
            paligemma_id=self.paligemma_id,
        )

        # Accelerate 저장 형식 로드 (CPU → 필터링 → GPU, MoNaVLA lean-loading 방식)
        bin_path = self.model_path / "pytorch_model.bin"
        sf_path  = self.model_path / "model.safetensors"
        if bin_path.exists():
            state = torch.load(bin_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=False)
            del state; gc.collect()
        elif sf_path.exists():
            from safetensors.torch import load_file
            state = load_file(str(sf_path), device="cpu")
            self.model.load_state_dict(state, strict=False)
            del state; gc.collect()
        else:
            print(f"[Engine] 경고: 가중치 파일 없음, 랜덤 가중치 사용")

        # INT8 모델은 device 이동 시 to()가 아닌 cuda() 사용 (BnB 요구사항)
        if self.use_int8:
            self.model.action_expert = self.model.action_expert.to(self.device)
            self.model.eval()
        else:
            self.model = self.model.to(self.device).eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 더미 추론 (CUDA 워밍업)
        dummy_img = np.zeros((self.IMG_SIZE, self.IMG_SIZE, 3), dtype=np.uint8)
        _ = self.predict(dummy_img, "warmup")
        print(f"[Engine] 준비 완료 (device={self.device}, solver={self.solver.__class__.__name__}, steps={self.n_ode_steps})")

    # ──────────────────────────────────────────
    # 이미지 전처리
    # ──────────────────────────────────────────

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Args:
            image: (H, W, 3) uint8 numpy
        Returns:
            tensor: (1, 1, 3, IMG_SIZE, IMG_SIZE) FP16 on device
        """
        # 학습(data/dataset.py)과 동일한 BILINEAR 보간 — 미지정 시 Pillow 기본값(BICUBIC)으로
        # 떨어져 train/inference 리사이즈 불일치 발생 (MoNaVLA에서 같은 클래스 버그 확인됨).
        pil = PILImage.fromarray(image).resize((self.IMG_SIZE, self.IMG_SIZE), PILImage.BILINEAR)
        arr = np.array(pil).transpose(2, 0, 1).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, H, W)
        return t.to(self.device)  # backbone이 자체 dtype으로 캐스팅

    def _image_hash(self, image: np.ndarray) -> str:
        return hashlib.md5(image.tobytes()).hexdigest()

    # ──────────────────────────────────────────
    # 비전 특징 인코딩 (캐시 지원)
    # ──────────────────────────────────────────

    @torch.no_grad()
    def encode_observation(self, image: np.ndarray, instruction: str) -> torch.Tensor:
        """
        이미지 + 텍스트 → VLM 조건 벡터 (B=1, n_tokens, dim)
        캐시 TTL 내 동일 이미지면 재계산 생략.
        """
        assert self.model is not None, "warmup()을 먼저 호출하세요"

        if self.cache_vision:
            h = self._image_hash(image)
            now = time.monotonic()
            if h in self._vision_cache:
                cond, ts = self._vision_cache[h]
                if now - ts < self._cache_ttl:
                    return cond

        img_t = self._preprocess_image(image)
        cond = self.model.forward_backbone(img_t, [instruction])

        if self.cache_vision:
            self._vision_cache[h] = (cond, time.monotonic())
            # 오래된 캐시 정리
            if len(self._vision_cache) > 10:
                oldest = min(self._vision_cache, key=lambda k: self._vision_cache[k][1])
                del self._vision_cache[oldest]

        return cond

    # ──────────────────────────────────────────
    # 액션 청크 샘플링
    # ──────────────────────────────────────────

    @torch.no_grad()
    def sample_actions(self, cond: torch.Tensor) -> np.ndarray:
        """
        Args:
            cond: (1, n_tokens, dim) FP16 조건 벡터
        Returns:
            actions: (horizon, action_dim) float32 numpy  [정규화 공간 -1~1]
        """
        B = cond.shape[0]
        x_init = torch.randn(B, self.horizon, self.action_dim, device=self.device, dtype=cond.dtype)

        def velocity_fn(x_t, t_batch, c):
            return self.model.action_expert(x_t, t_batch, c)

        x_final = self.solver.solve(velocity_fn, x_init, cond)
        return x_final.squeeze(0).float().cpu().numpy()  # (horizon, action_dim)

    # ──────────────────────────────────────────
    # 통합 진입점
    # ──────────────────────────────────────────

    def predict(self, image: np.ndarray, instruction: str) -> np.ndarray:
        """
        단일 이미지 + instruction → 역정규화된 속도 청크

        Args:
            image:       (H, W, 3) uint8 numpy
            instruction: 자연어 지시어 (9-class prefix 포함해도 무방)
        Returns:
            actions: (horizon, 3) float32 numpy  [vx(m/s), vy(m/s), wz(rad/s)]
        """
        cond = self.encode_observation(image, instruction)
        actions_norm = self.sample_actions(cond)
        actions_raw = self.normalizer.unnormalize(actions_norm)
        return actions_raw

    def predict_latency(self, image: np.ndarray, instruction: str) -> tuple[np.ndarray, float]:
        """predict() + latency 측정"""
        t0 = time.monotonic()
        actions = self.predict(image, instruction)
        latency_ms = (time.monotonic() - t0) * 1000
        return actions, latency_ms
