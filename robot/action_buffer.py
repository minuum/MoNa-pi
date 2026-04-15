"""
ActionChunkBuffer — 추론 서버에서 받은 액션 청크를 로컬에서 소비하는 버퍼

역할:
    - MoNa-pi 추론 서버는 2Hz 주기로 10-step 액션 청크를 생성
    - 로봇 제어 루프(10Hz)는 이 버퍼에서 1-step씩 꺼내 cmd_vel로 발행
    - 청크의 50%가 소진되면 재계획(replan) 신호 발생

Receding Horizon Control 흐름:
    t=0   : 추론 서버 → push([a0, a1, ..., a9])
    t=0.1 : 제어 루프 → pop() = a0
    t=0.2 :            pop() = a1
    ...
    t=0.5 :            pop() = a4  ← should_replan() = True
    → 추론 서버에 새 청크 요청
    t=0.6 :            pop() = a5  (이전 청크 계속 소비)
    ...
    t=1.0 : 새 청크 도착 → push([a0', ..., a9'])  → 자연스럽게 교체
"""

import threading
import time
from collections import deque

import numpy as np


class ActionChunkBuffer:
    """
    스레드 안전 액션 청크 버퍼.

    Args:
        horizon:      청크 크기 (기본 10)
        replan_ratio: 이 비율만큼 소진되면 should_replan() = True (기본 0.5)
        zero_on_empty: True이면 버퍼 비었을 때 zero 액션 반환, False이면 last 액션 유지
    """

    def __init__(self, horizon: int = 10, replan_ratio: float = 0.5, zero_on_empty: bool = True):
        self.horizon = horizon
        self.replan_ratio = replan_ratio
        self.zero_on_empty = zero_on_empty
        self.action_dim = 3  # (vx, vy, wz)

        self._lock = threading.Lock()
        self._buffer: deque[np.ndarray] = deque()
        self._total_consumed = 0
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._replan_triggered = False
        self._emergency = False

    # ──────────────────────────────────────────
    # 쓰기 (추론 서버 → 버퍼)
    # ──────────────────────────────────────────

    def push(self, actions: np.ndarray):
        """
        새 액션 청크를 버퍼에 주입.
        기존 잔여 액션 위에 덮어쓰기 (새 청크로 교체).

        Args:
            actions: (horizon, action_dim) float32 numpy
        """
        assert actions.shape == (self.horizon, self.action_dim), \
            f"청크 형태 불일치: {actions.shape} != ({self.horizon}, {self.action_dim})"

        with self._lock:
            self._buffer.clear()
            for step in actions:
                self._buffer.append(step.astype(np.float32))
            self._total_consumed = 0
            self._replan_triggered = False

    # ──────────────────────────────────────────
    # 읽기 (제어 루프 → 모터)
    # ──────────────────────────────────────────

    def pop(self) -> np.ndarray:
        """
        다음 액션 스텝 반환.
        비상 정지 상태이거나 버퍼가 비었으면 zero/last 액션 반환.

        Returns:
            action: (action_dim,) float32 numpy  [vx, vy, wz]
        """
        with self._lock:
            if self._emergency:
                return np.zeros(self.action_dim, dtype=np.float32)

            if len(self._buffer) == 0:
                if self.zero_on_empty:
                    return np.zeros(self.action_dim, dtype=np.float32)
                return self._last_action.copy()

            action = self._buffer.popleft()
            self._total_consumed += 1
            self._last_action = action.copy()

            # replan 트리거 확인
            replan_threshold = int(self.horizon * self.replan_ratio)
            if self._total_consumed >= replan_threshold and not self._replan_triggered:
                self._replan_triggered = True

            return action

    # ──────────────────────────────────────────
    # 상태 확인
    # ──────────────────────────────────────────

    def should_replan(self) -> bool:
        """청크 재계획이 필요한 시점이면 True. 한 번 True 반환 후 리셋."""
        with self._lock:
            if self._replan_triggered:
                self._replan_triggered = False
                return True
            return False

    def remaining(self) -> int:
        with self._lock:
            return len(self._buffer)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buffer) == 0

    # ──────────────────────────────────────────
    # 비상 정지
    # ──────────────────────────────────────────

    def emergency_stop(self):
        """즉시 버퍼 비우기 + 이후 pop()은 항상 zero 반환"""
        with self._lock:
            self._buffer.clear()
            self._emergency = True

    def resume(self):
        """비상 정지 해제"""
        with self._lock:
            self._emergency = False
            self._total_consumed = 0

    def is_emergency(self) -> bool:
        with self._lock:
            return self._emergency
