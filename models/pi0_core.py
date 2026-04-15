import torch
import torch.nn as nn
from .backbones.pi0_backbone import Pi0Backbone
from .heads.flow_head import FlowMatchingHead

class Pi0VLA(nn.Module):
    """
    MoNa-pi Main Model Wrapper (Original pi0 Spec)
    """
    def __init__(
        self,
        action_dim=3,
        horizon=10,
        hidden_dim=512,
        vision_model_id="google/siglip-so400m-patch14-384",
        lang_model_id="google/gemma-2b",
        **kwargs
    ):
        super().__init__()
        
        # 1. Real pi0 Backbone (SigLIP + Gemma + Resampler)
        self.backbone = Pi0Backbone(
            vision_model_id=vision_model_id,
            lang_model_id=lang_model_id
        )
        
        self.backbone_out_dim = self.backbone.lang_hidden_size # e.g., 2048
        
        # 2. Flow Matching Head
        self.flow_head = FlowMatchingHead(
            input_dim=self.backbone_out_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim
        )
        
        # 3. Ensure all components are in half precision for Jetson AGX
        self.half()

    def forward_backbone(self, images, instructions):
        """
        images: (B, Window, C, H, W)
        """
        # pi0-style: Pass images through SigLIP -> Resampler -> Gemma
        # instructions are handled inside backbone or as prefix
        cond = self.backbone(images, instructions)
        return cond # (B, 64, 2048)

    def compute_loss(self, images, instructions, actions_gt):
        """
        Training loss computation
        """
        cond = self.forward_backbone(images, instructions)
        loss = self.flow_head.get_loss(actions_gt, cond)
        return loss

    @torch.no_grad()
    def sample_actions(self, images, instructions, n_steps=5):
        """
        ODE Solver (Heun's method)를 이용한 액션 샘플링
        """
        device = images.device
        dtype = images.dtype
        cond = self.forward_backbone(images, instructions)
        B = cond.shape[0]
        
        # 1. 초기 노이즈 (x_0 ~ N(0, I))
        x_t = torch.randn(B, self.flow_head.horizon, self.flow_head.action_dim, device=device, dtype=dtype)
        
        # 2. Heun's Method Solver
        for i in range(n_steps):
            t_curr = (i / n_steps)
            t_next = ((i + 1) / n_steps)
            dt = t_next - t_curr
            
            t_curr_tensor = torch.full((B,), t_curr, device=device, dtype=dtype)
            v_t = self.flow_head(x_t, t_curr_tensor, cond)
            
            # Predict x_next
            x_next_approx = x_t + v_t * dt
            
            # Correction step
            t_next_tensor = torch.full((B,), t_next, device=device, dtype=dtype)
            v_t_next = self.flow_head(x_next_approx, t_next_tensor, cond)
            
            x_t = x_t + (v_t + v_t_next) * 0.5 * dt
            
        return x_t
