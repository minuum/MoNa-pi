"""
π0 Action Expert

실제 π0 논문의 Action Expert:
    PaliGemma VLM 전체 토큰 시퀀스에 cross-attend하는 독립 Flow Matching transformer.

기존 FlowMatchingHead와 차이:
    - 이름: FlowMatchingHead → ActionExpert (π0 논문 용어)
    - 입력 cond: 64 latent → VLM 전체 시퀀스 (T_vis + L 토큰)
    - 인터페이스: 동일 (forward, get_loss)

내부적으로 FlowMatchingHead를 재사용하여 코드 중복 최소화.
"""

import torch
import torch.nn as nn
from .flow_head import FlowMatchingHead


class ActionExpert(nn.Module):
    """
    π0 Action Expert

    Args:
        action_dim: 액션 차원 (기본 3: vx, vy, ωz)
        horizon:    예측 청크 크기 (기본 10)
        hidden_dim: Transformer 히든 차원 (기본 512)
        cond_dim:   VLM 토큰 차원 = Gemma hidden size (기본 2048)
        n_layers:   Transformer 레이어 수 (기본 4)
        n_heads:    Multi-head attention 헤드 수 (기본 8)
    """

    def __init__(
        self,
        action_dim: int = 3,
        horizon: int = 10,
        hidden_dim: int = 512,
        cond_dim: int = 2048,
        n_layers: int = 4,
        n_heads: int = 8,
    ):
        super().__init__()
        self._head = FlowMatchingHead(
            input_dim=cond_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
        )
        # 외부 접근용 속성 alias
        self.action_dim = action_dim
        self.horizon    = horizon

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t:  (B, horizon, action_dim)   — 현재 노이즈 액션
            t:    (B,) 또는 (B, 1)            — 현재 ODE 시간 [0, 1]
            cond: (B, T_vlm, cond_dim)        — PaliGemma VLM 토큰 시퀀스
        Returns:
            v_pred: (B, horizon, action_dim)  — 예측 velocity field
        """
        return self._head(x_t, t, cond)

    def get_loss(self, x_1: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Conditional Flow Matching 학습 손실

        Args:
            x_1:  (B, horizon, action_dim)  — ground truth 액션 (noise-free)
            cond: (B, T_vlm, cond_dim)      — PaliGemma VLM 토큰 시퀀스
        Returns:
            loss: scalar — MSE(v_pred, v_target)
        """
        return self._head.get_loss(x_1, cond)
