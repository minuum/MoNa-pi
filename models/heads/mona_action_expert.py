import torch
import torch.nn as nn
from .flow_head import FlowMatchingHead

class MoNaActionExpert(nn.Module):
    """
    MoNa-pi Project Main Action Expert
    
    π0 정통 아키텍처를 계승하며 MoNa-pi v5 데이터셋 특성에 최적화된 버전.
    연구용 베이스라인(ActionExpert)과 분리하여 실제 주행 학습에 사용함.
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
        self.action_dim = action_dim
        self.horizon = horizon
        
        # MoNa-pi v5 데이터셋(1.150)을 위한 기본 정규화 파라미터 
        # (π0 정통에서도 수치 안정성을 위해 필수적인 처리)
        self.register_buffer("mean", torch.zeros(action_dim))
        self.register_buffer("std", torch.ones(action_dim) * 1.2) # v5 vx, vy max가 대략 1.15

        self._head = FlowMatchingHead(
            input_dim=cond_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
        )

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        """Raw 물리값(1.150 등) -> 정규화 공간(~1.0) 변환"""
        return (actions - self.mean) / (self.std + 1e-6)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """ODE Solver에서 호출되는 Velocity Field 예측"""
        return self._head(x_t, t, cond)

    def get_loss(self, actions_gt: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        MoNa-pi v5 GT 연산을 위한 손실 함수.
        """
        # [Normalization] 1.150 -> ~1.0 범위로 변환
        normalized_gt = self.normalize(actions_gt)
        
        return self._head.get_loss(normalized_gt, cond)

    @torch.no_grad()
    def unnormalize(self, x_t: torch.Tensor) -> torch.Tensor:
        """모델이 생성한 노이즈(~1.0)를 다시 로봇 물리값(1.150)으로 복원"""
        return x_t * self.std + self.mean
