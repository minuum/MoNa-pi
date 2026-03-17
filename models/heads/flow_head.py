import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class FlowMatchingHead(nn.Module):
    """
    Advanced Flow Matching Action Head (MoNa-pi)
    
    특징:
    - Sinusoidal Time Embedding
    - Cross-Attention for VLM Condition
    - Transformer Encoder for Action Sequence Modeling
    """
    def __init__(
        self,
        input_dim: int,
        action_dim: int = 3,         # linear_x, linear_y, angular_z
        horizon: int = 10,           # V3 Standard Horizon (k=10)
        hidden_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        
        # 1. Action & Time Embeddings
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 2. Condition Projection
        self.cond_proj = nn.Linear(input_dim, hidden_dim)
        
        # 3. Transformer Blocks with Cross-Attention
        # 간단한 구현을 위해 nn.MultiheadAttention과 Residual 사용
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True),
                'cross_attn': nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                ),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
                'norm3': nn.LayerNorm(hidden_dim)
            }) for _ in range(n_layers)
        ])
        
        # 4. Output Head
        self.output_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x_t, t, cond):
        """
        Args:
            x_t: (B, horizon, action_dim)
            t: (B, 1) or (B,)
            cond: (B, n_tokens, input_dim)
        """
        B, H, D = x_t.shape
        
        # Embed Time & Action
        t_emb = self.time_mlp(t.squeeze(-1)).unsqueeze(1) # (B, 1, hidden_dim)
        h = self.action_proj(x_t) + t_emb # (B, H, hidden_dim)
        
        # Embed Condition
        c = self.cond_proj(cond) # (B, n_tokens, hidden_dim)
        
        # Transformer Layers
        for layer in self.layers:
            # Self-Attention on Action Sequence
            attn_out, _ = layer['self_attn'](layer['norm1'](h), h, h)
            h = h + attn_out
            
            # Cross-Attention on VLM Features
            cross_out, _ = layer['cross_attn'](layer['norm2'](h), c, c)
            h = h + cross_out
            
            # MLP
            h = h + layer['mlp'](layer['norm3'](h))
            
        return self.output_head(h) # (B, H, D)

    def get_loss(self, x_1, cond):
        """
        Flow Matching Loss (Conditional Flow Matching)
        """
        B = x_1.shape[0]
        device = x_1.device
        
        t = torch.rand(B, 1, device=device)
        x_0 = torch.randn_like(x_1)
        
        # x_t = (1-t)x_0 + t*x_1
        t_expand = t.view(B, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        
        v_target = x_1 - x_0
        v_pred = self.forward(x_t, t, cond)
        
        return F.mse_loss(v_pred, v_target)
