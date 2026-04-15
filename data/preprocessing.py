"""
MoNa-pi 데이터 전처리 유틸리티

파이프라인 순서:
    raw actions → ActionSmoother → ActionNormalizer → IntentPrefixInjector

MoNaVLA 계승 항목:
    - CLIP 이미지 정규화 상수 (mean/std)
    - CounterfactualInjector  (stop/steer 오버라이드)
    - 다국어 action-aware instruction (한/영, 20% 노이즈)
    - HFlip 증강 시 left↔right instruction 교체 로직
"""

import random
import numpy as np
import torch
from scipy.signal import savgol_filter

# ─── MoNaVLA 에서 직접 이관: CLIP/SigLIP 이미지 정규화 상수 ────────────────
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


# ─────────────────────────────────────────────
# 1. ActionSmoother — Savitzky-Golay 필터
#    키보드 Bang-bang 스파이크 제거
# ─────────────────────────────────────────────

class ActionSmoother:
    """
    Savitzky-Golay 필터로 액션 시퀀스를 부드럽게 만든다.

    - window_length: 홀수여야 함. 에피소드가 짧을 경우 자동으로 줄임.
    - polyorder: 다항식 차수 (window_length 보다 작아야 함)
    """

    def __init__(self, window_length: int = 11, polyorder: int = 3):
        self.window_length = window_length
        self.polyorder = polyorder

    def smooth(self, actions: np.ndarray) -> np.ndarray:
        """
        Args:
            actions: (T, action_dim) numpy array
        Returns:
            smoothed: (T, action_dim) numpy array
        """
        T = actions.shape[0]
        if T < self.polyorder + 2:
            return actions.copy()

        # 에피소드 길이에 맞게 window 조정 (홀수 유지)
        win = min(self.window_length, T if T % 2 == 1 else T - 1)
        win = max(win, self.polyorder + 2 if (self.polyorder + 2) % 2 == 1 else self.polyorder + 3)

        smoothed = np.zeros_like(actions)
        for dim in range(actions.shape[1]):
            smoothed[:, dim] = savgol_filter(actions[:, dim], win, self.polyorder)
        return smoothed

    def smooth_tensor(self, actions: torch.Tensor) -> torch.Tensor:
        arr = actions.numpy() if not actions.is_cuda else actions.cpu().numpy()
        return torch.from_numpy(self.smooth(arr)).to(actions.device)


# ─────────────────────────────────────────────
# 2. ActionNormalizer — [-1.15, 1.15] → [-1, 1]
#    기존 dataset.py ActionNormalizer 대체 버전
# ─────────────────────────────────────────────

class ActionNormalizer:
    """
    3-DOF 옴니휠 로봇 액션 정규화 (lx, ly, az)
    기본 범위: [-1.15, 1.15]
    """

    def __init__(self, min_val: float = -1.15, max_val: float = 1.15, action_dim: int = 3):
        self.min_vals = np.full(action_dim, min_val, dtype=np.float32)
        self.max_vals = np.full(action_dim, max_val, dtype=np.float32)

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        """[min, max] → [-1, 1]"""
        return 2.0 * (actions - self.min_vals) / (self.max_vals - self.min_vals + 1e-8) - 1.0

    def unnormalize(self, actions_norm: np.ndarray) -> np.ndarray:
        """[-1, 1] → [min, max]"""
        return 0.5 * (actions_norm + 1.0) * (self.max_vals - self.min_vals) + self.min_vals

    def normalize_tensor(self, actions: torch.Tensor) -> torch.Tensor:
        min_t = torch.from_numpy(self.min_vals).to(actions.device)
        max_t = torch.from_numpy(self.max_vals).to(actions.device)
        return 2.0 * (actions - min_t) / (max_t - min_t + 1e-8) - 1.0

    def unnormalize_tensor(self, actions: torch.Tensor) -> torch.Tensor:
        min_t = torch.from_numpy(self.min_vals).to(actions.device)
        max_t = torch.from_numpy(self.max_vals).to(actions.device)
        return 0.5 * (actions + 1.0) * (max_t - min_t) + min_t


# ─────────────────────────────────────────────
# 3. IntentPrefixInjector — 9-class 의도 자동 분류
#    MobileVLA-R1 방식: language instruction 앞에 의도 태그 접두사 추가
#    MoNaVLA 계승: 20% 노이즈 + 한/영 다국어 다변형
# ─────────────────────────────────────────────

# 9-class 의도 레이블
INTENT_LABELS = [
    "FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT",
    "STRAFE_LEFT", "STRAFE_RIGHT", "FORWARD_LEFT", "FORWARD_RIGHT", "STOP",
]

# MoNaVLA 에서 이관: 의도별 한/영 다국어 instruction 변형
_INTENT_VARIATIONS: dict[str, list[str]] = {
    "STOP": [
        "Halt in front of the object", "Freeze and stay still", "Do not move",
        "Stand by at current position", "Maintain pose near the basket",
        "바구니 앞에서 멈춰", "움직임을 중단하고 대기해", "현재 위치에서 정지",
    ],
    "FORWARD": [
        "Direct route to the gray basket", "Straight ahead to the target",
        "Proceed front toward the object", "Navigate straight",
        "바구니를 향해 쭉 직진해", "정면 목표로 전진", "방향 꺾지 말고 그대로 가",
    ],
    "BACKWARD": [
        "Move backward slowly", "Reverse direction",
        "뒤로 천천히 후진해", "후진해",
    ],
    "TURN_LEFT": [
        "Rotate left toward the basket", "Spin left to see the target",
        "Steer toward the left side", "Left turn required",
        "좌측으로 회전해", "왼쪽으로 각도를 틀어", "왼쪽 방향으로 보정해",
    ],
    "TURN_RIGHT": [
        "Rotate right toward the basket", "Spin right to see the target",
        "Steer toward the right side", "Right turn required",
        "우측으로 회전해", "오른쪽으로 조향을 바꿔", "우측으로 움직여",
    ],
    "STRAFE_LEFT": [
        "Slide left while facing forward", "Move laterally to the left",
        "왼쪽으로 옆으로 이동해", "측면 이동 왼쪽",
    ],
    "STRAFE_RIGHT": [
        "Slide right while facing forward", "Move laterally to the right",
        "오른쪽으로 옆으로 이동해", "측면 이동 오른쪽",
    ],
    "FORWARD_LEFT": [
        "Angle toward left-front side", "Diagonal path to the left",
        "Shift left while moving forward", "Bear left of the basket",
        "왼쪽 대각선으로 비스듬히 접근해", "전진하면서 왼쪽으로 살짝 틀어",
    ],
    "FORWARD_RIGHT": [
        "Angle toward right-front side", "Diagonal path to the right",
        "Shift right while moving forward", "Bear right of the basket",
        "오른쪽 대각선으로 비스듬히 가", "전진하면서 조향을 오른쪽으로 살짝 유지해",
    ],
}

# 20% 노이즈용 범용 instruction (MoNaVLA 계승)
_GENERIC_VARIATIONS = [
    "Navigate to the gray basket",
    "Go to the target object",
    "Proceed to the destination",
    "바구니가 보일 때까지 계속 이동해",
    "목표물을 향해 가줘",
    "앞에 보이는 목표물로 이동해",
]

# HFlip 증강 시 left↔right 교체 맵 (MoNaVLA 계승)
_FLIP_LANG_MAP = {
    "left": "right", "right": "left",
    "좌측": "우측", "우측": "좌측",
    "왼쪽": "오른쪽", "오른쪽": "왼쪽",
    "TURN_LEFT": "TURN_RIGHT", "TURN_RIGHT": "TURN_LEFT",
    "STRAFE_LEFT": "STRAFE_RIGHT", "STRAFE_RIGHT": "STRAFE_LEFT",
    "FORWARD_LEFT": "FORWARD_RIGHT", "FORWARD_RIGHT": "FORWARD_LEFT",
}


class IntentPrefixInjector:
    """
    액션 청크의 주 방향을 분석하여 9-class 의도 태그 + 다국어 instruction을 생성.

    - 20% 확률로 generic instruction → 모델이 이미지에 의존하도록 강제
    - 나머지 80%: 의도에 맞는 한/영 다변형 instruction 선택
    - inject() 결과: "[INTENT] instruction"
    """

    MOVE_THRESH = 0.15
    ROT_THRESH  = 0.15
    NOISE_PROB  = 0.2  # MoNaVLA 동일

    def classify_chunk(self, actions: np.ndarray) -> str:
        mean = np.mean(actions, axis=0)
        lx, ly, az = mean[0], mean[1], mean[2]

        moving_fwd  = lx >  self.MOVE_THRESH
        moving_back = lx < -self.MOVE_THRESH
        strafing_r  = ly >  self.MOVE_THRESH
        strafing_l  = ly < -self.MOVE_THRESH
        turning_r   = az >  self.ROT_THRESH
        turning_l   = az < -self.ROT_THRESH

        if not any([moving_fwd, moving_back, strafing_r, strafing_l, turning_r, turning_l]):
            return "STOP"
        if moving_fwd and turning_l:
            return "FORWARD_LEFT"
        if moving_fwd and turning_r:
            return "FORWARD_RIGHT"
        if moving_fwd:
            return "FORWARD"
        if moving_back:
            return "BACKWARD"
        if turning_l:
            return "TURN_LEFT"
        if turning_r:
            return "TURN_RIGHT"
        if strafing_l:
            return "STRAFE_LEFT"
        if strafing_r:
            return "STRAFE_RIGHT"
        return "FORWARD"

    def inject(self, instruction: str, actions: np.ndarray, is_training: bool = True) -> str:
        """
        Returns:
            "[INTENT] {한/영 다변형 또는 generic instruction}"
        """
        intent = self.classify_chunk(actions)

        if is_training and random.random() < self.NOISE_PROB:
            # 20% 노이즈: 시각 의존성 강제
            body = random.choice(_GENERIC_VARIATIONS)
        else:
            variations = _INTENT_VARIATIONS.get(intent, [instruction])
            body = random.choice(variations)

        return f"[{intent}] {body}"

    @staticmethod
    def flip_instruction(instruction: str) -> str:
        """HFlip 증강 시 instruction에서 left↔right 교체 (MoNaVLA 이관)"""
        for k, v in _FLIP_LANG_MAP.items():
            if k in instruction:
                rev_v = _FLIP_LANG_MAP.get(v, k)
                instruction = instruction.replace(k, "__TEMP__").replace(v, k).replace("__TEMP__", v)
                break
        return instruction


# ─────────────────────────────────────────────
# 4. CounterfactualInjector — stop/steer 오버라이드
#    MoNaVLA 계승: 명령어 감수성 강화를 위한 counterfactual 학습
# ─────────────────────────────────────────────

_CF_STOP_VARS = [
    "Stop in front of the gray basket", "Halt immediately",
    "Freeze and stay still", "Do not move",
    "정지해", "움직이지 마", "멈춰",
]
_CF_TURN_LEFT_VARS  = ["Turn left", "Steer to the left", "왼쪽으로 가", "좌측으로 회전해"]
_CF_TURN_RIGHT_VARS = ["Turn right", "Steer to the right", "오른쪽으로 가", "우측으로 회전해"]


class CounterfactualInjector:
    """
    학습 중 일정 확률로 액션을 강제 오버라이드하여
    모델이 instruction을 무시하지 않도록 훈련.

    MoNaVLA 방식 완전 이관:
    - stop_prob:  이 확률로 전체 액션을 0으로, instruction을 "정지" 변형으로
    - steer_prob: 이 확률로 강한 좌/우 회전 액션 + 대응 instruction
    """

    def __init__(self, stop_prob: float = 0.05, steer_prob: float = 0.05):
        self.stop_prob = stop_prob
        self.steer_prob = steer_prob

    def apply(
        self,
        actions: np.ndarray,
        instruction: str,
        is_training: bool = True,
    ) -> tuple[np.ndarray, str]:
        """
        Args:
            actions:     (T, action_dim) numpy
            instruction: 원본 지시어
        Returns:
            (modified_actions, modified_instruction)
        """
        if not is_training:
            return actions, instruction

        if random.random() < self.stop_prob:
            return np.zeros_like(actions), f"[STOP] {random.choice(_CF_STOP_VARS)}"

        if random.random() < self.steer_prob:
            is_left = random.random() < 0.5
            if is_left:
                turn = np.zeros_like(actions)
                turn[:, 2] = 0.6   # angular_z → left
                return turn, f"[TURN_LEFT] {random.choice(_CF_TURN_LEFT_VARS)}"
            else:
                turn = np.zeros_like(actions)
                turn[:, 2] = -0.6  # angular_z → right
                return turn, f"[TURN_RIGHT] {random.choice(_CF_TURN_RIGHT_VARS)}"

        return actions, instruction


# ─────────────────────────────────────────────
# 5. DeltaActionConverter — 속도 → 변위 (옵션)
#    v = (Δx, Δy, Δθ) / dt 역산
# ─────────────────────────────────────────────

class DeltaActionConverter:
    """
    속도 명령(v_x, v_y, ω_z)을 dt 간격의 변위(Δx, Δy, Δθ)로 변환.
    Flow Matching이 절대 속도보다 변위를 더 안정적으로 학습한다는 연구 결과 반영.
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt  # 제어 주기 (초)

    def to_delta(self, actions: np.ndarray) -> np.ndarray:
        """velocity * dt = displacement"""
        return actions * self.dt

    def to_velocity(self, deltas: np.ndarray) -> np.ndarray:
        """displacement / dt = velocity"""
        return deltas / self.dt


# ─────────────────────────────────────────────
# 6. EpisodePreprocessor — 위 모듈들을 묶는 파이프라인
# ─────────────────────────────────────────────

class EpisodePreprocessor:
    """
    에피소드 단위로 전처리를 수행하는 통합 파이프라인.

    순서 (에피소드 단위):
        raw_actions → DeltaConverter → ActionSmoother

    순서 (청크 단위, __getitem__ 내부):
        CounterfactualInjector → ActionNormalizer → IntentPrefixInjector
    """

    def __init__(
        self,
        smooth: bool = True,
        normalize: bool = True,
        inject_prefix: bool = True,
        use_delta: bool = False,
        use_counterfactual: bool = False,
        smoother_window: int = 11,
        smoother_poly: int = 3,
        norm_min: float = -1.15,
        norm_max: float = 1.15,
        action_dim: int = 3,
        dt: float = 0.1,
        cf_stop_prob: float = 0.05,
        cf_steer_prob: float = 0.05,
    ):
        self.smoother = ActionSmoother(smoother_window, smoother_poly) if smooth else None
        self.normalizer = ActionNormalizer(norm_min, norm_max, action_dim) if normalize else None
        self.injector = IntentPrefixInjector() if inject_prefix else None
        self.converter = DeltaActionConverter(dt) if use_delta else None
        self.cf_injector = CounterfactualInjector(cf_stop_prob, cf_steer_prob) if use_counterfactual else None

    # ── 에피소드 단위 처리 ──────────────────────────────────────
    def process_episode_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        에피소드 전체 액션 배열에 스무딩/변환 적용.
        Args:
            actions: (T, action_dim) raw numpy
        Returns:
            processed: (T, action_dim) numpy
        """
        if self.converter is not None:
            actions = self.converter.to_delta(actions)
        if self.smoother is not None:
            actions = self.smoother.smooth(actions)
        return actions

    # ── 청크 단위 처리 ──────────────────────────────────────────
    def apply_counterfactual(
        self, actions: np.ndarray, instruction: str, is_training: bool = True
    ) -> tuple[np.ndarray, str]:
        """Counterfactual 주입 (stop/steer 오버라이드)"""
        if self.cf_injector is not None:
            return self.cf_injector.apply(actions, instruction, is_training)
        return actions, instruction

    def normalize_chunk(self, actions: np.ndarray) -> np.ndarray:
        if self.normalizer is not None:
            return self.normalizer.normalize(actions)
        return actions

    def normalize_chunk_tensor(self, actions: torch.Tensor) -> torch.Tensor:
        if self.normalizer is not None:
            return self.normalizer.normalize_tensor(actions)
        return actions

    def unnormalize_chunk_tensor(self, actions: torch.Tensor) -> torch.Tensor:
        if self.normalizer is not None:
            return self.normalizer.unnormalize_tensor(actions)
        return actions

    def get_instruction(
        self, instruction: str, actions: np.ndarray, is_training: bool = True
    ) -> str:
        """청크 액션을 보고 intent prefix + 다국어 instruction 생성"""
        if self.injector is not None:
            return self.injector.inject(instruction, actions, is_training)
        return instruction

    @staticmethod
    def flip_instruction(instruction: str) -> str:
        """HFlip 증강 시 instruction left↔right 교체"""
        return IntentPrefixInjector.flip_instruction(instruction)
