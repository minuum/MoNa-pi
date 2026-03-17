import torch
import torch.nn as nn
from .heads.flow_head import FlowMatchingHead

class Pi0VLA(nn.Module):
    """
    MoNa-pi Main Model Wrapper
    """
    def __init__(
        self,
        backbone,           # Pre-trained VLM
        resampler=None,     # Optional Perceiver Resampler
        action_dim=3,
        horizon=10,         # Chunk size 10 (99% certainty)
        hidden_dim=512,
        **kwargs
    ):
        super().__init__()
        self.backbone = backbone
        self.resampler = resampler
        
        backbone_out_dim = kwargs.get("backbone_out_dim", 1024)
        
        self.flow_head = FlowMatchingHead(
            input_dim=backbone_out_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim
        )

    def forward_backbone(self, images, instructions):
        """
        VLM Backbone을 통해 시각-언어 레이턴트 추출 (e.g. Kosmos-2)
        images: (B, Window, C, H, W)
        """
        B, W, C, H, W_img = images.shape
        # TODO: 실제 백본 통합 로직 (e.g. 8개 영상을 각각 혹은 통합하여 처리)
        # 현재는 Placeholder로 64개의 토큰(Perceiver Resampler 결과 형태) 반환
        device = images.device
        return torch.randn(B, 64, self.flow_head.input_dim, device=device)

    def compute_loss(self, images, instructions, actions_gt):
        """
        Training loss 계산
        """
        cond = self.forward_backbone(images, instructions)
        loss = self.flow_head.get_loss(actions_gt, cond)
        return loss

    @torch.no_grad()
    def sample_actions(self, images, instructions, n_steps=5, solver='heun'):
        """
        Inference: ODE Solver (Euler or Heun)를 사용한 액션 샘플링
        """
        B = images.shape[0]
        device = images.device
        horizon = self.flow_head.horizon
        action_dim = self.flow_head.action_dim
        
        # 1. Feature Extraction
        cond = self.forward_backbone(images, instructions)
        
        # 2. Noise Initial State x_0 ~ N(0, I)
        x_t = torch.randn(B, horizon, action_dim, device=device)
        
        # 3. ODE Integration
        dt = 1.0 / n_steps
        
        for i in range(n_steps):
            t_curr = (i / n_steps)
            t = torch.ones(B, 1, device=device) * t_curr
            
            if solver == 'euler':
                v_t = self.flow_head(x_t, t, cond)
                x_t = x_t + v_t * dt
            elif solver == 'heun':
                # Heun's method (2nd order ODE solver)
                v_t = self.flow_head(x_t, t, cond)
                
                t_next = (i + 1) / n_steps
                t_next_tensor = torch.ones(B, 1, device=device) * t_next
                
                x_next_euler = x_t + v_t * dt
                v_t_next = self.flow_head(x_next_euler, t_next_tensor, cond)
                
                x_t = x_t + 0.5 * (v_t + v_t_next) * dt
            else:
                raise ValueError(f"Unknown solver: {solver}")
            
        return x_t # Final actions x_1
