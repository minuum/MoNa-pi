"""
Flow Matching ODE 솔버 모음

사용 목적:
    Flow Matching 모델의 추론 시 노이즈 x_0 → 액션 x_1 을
    ODE를 풀어서 샘플링한다.

지원 솔버:
    - EulerSolver  : 가장 빠름, 정확도 낮음 (1-3 steps)
    - HeunSolver   : 속도/정확도 균형 (5-10 steps) ← 기본 권장
    - DPMSolver    : 3 steps에서 고품질 (실험적)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Callable


# ─────────────────────────────────────────────
# 타입 정의
# ─────────────────────────────────────────────

# velocity_fn(x_t, t_batch, cond) → (B, horizon, action_dim)
VelocityFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


# ─────────────────────────────────────────────
# Euler Solver
# ─────────────────────────────────────────────

class EulerSolver:
    """
    1차 Euler 적분.
    n_steps=1~3 으로도 동작하나 정확도 낮음.
    """

    def __init__(self, n_steps: int = 5):
        self.n_steps = n_steps

    def solve(
        self,
        velocity_fn: VelocityFn,
        x_init: torch.Tensor,     # (B, horizon, action_dim) 노이즈
        cond: torch.Tensor,       # (B, n_tokens, dim) VLM 조건
    ) -> torch.Tensor:
        x = x_init
        ts = torch.linspace(0.0, 1.0, self.n_steps + 1, device=x.device)
        for i in range(self.n_steps):
            t0 = ts[i]
            dt = ts[i + 1] - ts[i]
            t_batch = t0.expand(x.shape[0])
            v = velocity_fn(x, t_batch, cond)
            x = x + dt * v
        return x


# ─────────────────────────────────────────────
# Heun Solver (2차 Runge-Kutta)
# ─────────────────────────────────────────────

class HeunSolver:
    """
    Heun's method (Improved Euler / RK2).
    pi0_core.py 의 기존 Heun 구현을 독립 클래스로 이관.

    n_steps=5 이면 실시간 추론(< 200ms on Jetson AGX) 가능.
    """

    def __init__(self, n_steps: int = 5):
        self.n_steps = n_steps

    def solve(
        self,
        velocity_fn: VelocityFn,
        x_init: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        x = x_init
        ts = torch.linspace(0.0, 1.0, self.n_steps + 1, device=x.device)
        for i in range(self.n_steps):
            t0 = ts[i]
            t1 = ts[i + 1]
            dt = t1 - t0
            t0_batch = t0.expand(x.shape[0])
            t1_batch = t1.expand(x.shape[0])

            k1 = velocity_fn(x, t0_batch, cond)
            x_pred = x + dt * k1
            k2 = velocity_fn(x_pred, t1_batch, cond)
            x = x + dt * 0.5 * (k1 + k2)
        return x


# ─────────────────────────────────────────────
# DPM-Solver (3rd order, 3 steps)
# ─────────────────────────────────────────────

class DPMSolver:
    """
    DPM-Solver++ 에서 아이디어를 빌린 단순화된 고차 솔버.
    3 steps 만으로도 Heun 5 steps 에 근접한 품질.
    (정식 DPM-Solver는 diffusers 라이브러리 참조)

    구현은 3rd-order Adams–Bashforth 에 기반.
    """

    def __init__(self, n_steps: int = 3):
        self.n_steps = n_steps

    def solve(
        self,
        velocity_fn: VelocityFn,
        x_init: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        x = x_init
        ts = torch.linspace(0.0, 1.0, self.n_steps + 1, device=x.device)
        history = []

        for i in range(self.n_steps):
            t0 = ts[i]
            t1 = ts[i + 1]
            dt = t1 - t0
            t0_batch = t0.expand(x.shape[0])

            k = velocity_fn(x, t0_batch, cond)
            history.append(k)

            if len(history) == 1:
                x = x + dt * history[0]
            elif len(history) == 2:
                x = x + dt * (1.5 * history[-1] - 0.5 * history[-2])
            else:
                x = x + dt * ((23 * history[-1] - 16 * history[-2] + 5 * history[-3]) / 12)

        return x


# ─────────────────────────────────────────────
# 솔버 팩토리
# ─────────────────────────────────────────────

def build_solver(name: str, n_steps: int = 5):
    """
    Args:
        name: "euler" | "heun" | "dpm"
        n_steps: ODE 적분 스텝 수
    """
    name = name.lower()
    if name == "euler":
        return EulerSolver(n_steps)
    elif name == "heun":
        return HeunSolver(n_steps)
    elif name == "dpm":
        return DPMSolver(n_steps)
    raise ValueError(f"알 수 없는 솔버: {name}")
