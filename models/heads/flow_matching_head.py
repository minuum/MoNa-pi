import torch
import torch.nn as nn

from .flow_head import FlowMatchingHead


class FlowMatchingActionHead(nn.Module):
    """
    Skeleton wrapper for Flow Matching action head integration.

    This provides a stable interface for training/inference while
    deferring the core architecture to FlowMatchingHead.
    """

    def __init__(
        self,
        cond_dim: int,
        action_dim: int = 2,
        horizon: int = 10,
        hidden_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.action_dim = action_dim
        self.horizon = horizon

        self.core = FlowMatchingHead(
            input_dim=cond_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: (B, horizon, action_dim) noisy/partial action trajectory
            t: (B,) or (B, 1) time scalar for Flow Matching
            cond: (B, n_tokens, cond_dim) conditioning features from VLM
        Returns:
            v_pred: (B, horizon, action_dim) predicted velocity field
        """
        return self.core(x_t, t, cond)

    def loss(self, x_1: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Wrapper around Flow Matching loss.
        TODO: extend to log auxiliary metrics (e.g., MSE/PM/DM) for comparison.
        """
        return self.core.get_loss(x_1, cond)

    def sample(self, cond: torch.Tensor, steps: int = 50) -> torch.Tensor:
        """
        TODO: implement a sampler/ODE solver for inference.
        This placeholder exists to define the interface.
        """
        raise NotImplementedError("Flow Matching sampler not implemented yet")
