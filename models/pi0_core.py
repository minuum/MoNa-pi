import torch
import torch.nn as nn
from .heads.flow_head import FlowMatchingHead


class Pi0VLA(nn.Module):
    """
    MoNa-pi main model.

    backbone: PaliGemmaBackbone 인스턴스 (또는 None → placeholder 사용)
    """
    def __init__(
        self,
        backbone=None,
        action_dim: int = 3,
        horizon: int = 10,
        hidden_dim: int = 512,
        backbone_out_dim: int = 1024,
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self._backbone_out_dim = backbone_out_dim

        self.flow_head = FlowMatchingHead(
            input_dim=backbone_out_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
        )

    def forward_backbone(self, images, instructions):
        """
        Returns: cond (B, n_tokens, backbone_out_dim)
        """
        if self.backbone is not None:
            return self.backbone(images, instructions)

        # placeholder — 실제 백본 없을 때 테스트용
        B = images.shape[0]
        device = images.device
        return torch.randn(B, 64, self._backbone_out_dim, device=device)

    def compute_loss(self, images, instructions, actions_gt):
        cond = self.forward_backbone(images, instructions)
        return self.flow_head.get_loss(actions_gt, cond)

    @torch.no_grad()
    def sample_actions(self, images, instructions, n_steps=5, solver='heun'):
        """
        Returns: actions (B, horizon, action_dim)
        """
        B = images.shape[0]
        device = images.device
        horizon = self.flow_head.horizon
        action_dim = self.flow_head.action_dim

        cond = self.forward_backbone(images, instructions)
        x_t = torch.randn(B, horizon, action_dim, device=device)

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_curr = torch.ones(B, 1, device=device) * (i / n_steps)

            if solver == 'euler':
                x_t = x_t + self.flow_head(x_t, t_curr, cond) * dt

            elif solver == 'heun':
                v0 = self.flow_head(x_t, t_curr, cond)
                x_euler = x_t + v0 * dt
                t_next = torch.ones(B, 1, device=device) * ((i + 1) / n_steps)
                v1 = self.flow_head(x_euler, t_next, cond)
                x_t = x_t + 0.5 * (v0 + v1) * dt

            else:
                raise ValueError(f"Unknown solver: {solver}")

        return x_t
