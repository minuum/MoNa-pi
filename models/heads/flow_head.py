import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. Timestep Embedder ──────────────────────────────────────────────────────

class TimestepEmbedder(nn.Module):
    """
    Sinusoidal → MLP → (B, hidden_dim)
    π0 논문 방식: DiT 동일
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) → (B, hidden_dim)
        half = self.hidden_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
        )
        args = t[:, None] * freqs[None]          # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, hidden_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B,1) — 모델 dtype으로 캐스팅 (float32 입력 허용)
        if t.ndim == 2:
            t = t.squeeze(-1)
        dtype = next(self.mlp.parameters()).dtype
        return self.mlp(self._sinusoidal(t.to(dtype)))  # (B, hidden_dim)


# ── 2. AdaLN Modulation ───────────────────────────────────────────────────────

class AdaLNModulation(nn.Module):
    """
    cond_emb → (α1, β1, γ1, α2, β2, γ2)  각 (B, 1, hidden_dim)

    Zero-init: 학습 초기 identity 보장 (DiT / π0 논문 방식)
    self-attn + mlp 두 블록에 대한 scale/shift/gate
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_dim, 6 * hidden_dim)
        # Zero-init
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond_emb: torch.Tensor):
        # cond_emb: (B, hidden_dim)
        out = self.linear(self.silu(cond_emb))         # (B, 6*hidden_dim)
        chunks = out.chunk(6, dim=-1)                  # 6 × (B, hidden_dim)
        return tuple(c.unsqueeze(1) for c in chunks)   # 6 × (B, 1, hidden_dim)


# ── 3. FlowMatchingHead (π0 AdaLN-Zero) ──────────────────────────────────────

class FlowMatchingHead(nn.Module):
    """
    π0 Action Expert — AdaLN-Zero 방식 Flow Matching Transformer

    구조:
        action_proj  : Linear(action_dim → hidden_dim)
        timestep_emb : TimestepEmbedder  → cond_emb (B, hidden_dim)
        cond_proj    : Linear(cond_dim → hidden_dim)  — VLM 토큰 차원 축소

        N × Transformer Block (AdaLN-Zero):
            self_attn  : action tokens 내부  + AdaLN (α1,β1,γ1)
            cross_attn : action → VLM cond  (timestep 미적용)
            mlp        : FFN                + AdaLN (α2,β2,γ2)

        output_head  : Linear(hidden_dim → action_dim)
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 3,
        horizon: int = 10,
        hidden_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon    = horizon
        self.hidden_dim = hidden_dim

        # Embeddings
        self.action_proj    = nn.Linear(action_dim, hidden_dim)
        self.timestep_emb   = TimestepEmbedder(hidden_dim)
        self.cond_proj      = nn.Linear(input_dim, hidden_dim)

        # Per-layer modules
        self.self_attn_list  = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.cross_attn_list = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.mlp_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            for _ in range(n_layers)
        ])
        self.norm1_list = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.norm2_list = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.norm3_list = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.adaLN_mods = nn.ModuleList([AdaLNModulation(hidden_dim) for _ in range(n_layers)])

        # Output
        self.output_head = nn.Linear(hidden_dim, action_dim)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t  : (B, horizon, action_dim)
            t    : (B,) or (B, 1)
            cond : (B, T_vlm, input_dim)
        Returns:
            v_pred : (B, horizon, action_dim)
        """
        # 1. Embed
        h         = self.action_proj(x_t)           # (B, H, hidden_dim)
        cond_emb  = self.timestep_emb(t)             # (B, hidden_dim)
        c         = self.cond_proj(cond)              # (B, T_vlm, hidden_dim)

        # 2. Transformer blocks
        for i in range(len(self.self_attn_list)):
            α1, β1, γ1, α2, β2, γ2 = self.adaLN_mods[i](cond_emb)

            # Self-Attention (AdaLN-Zero)
            q = self.norm1_list[i](h) * (1 + α1) + β1
            attn_out, _ = self.self_attn_list[i](q, q, q)
            h = h + γ1 * attn_out

            # Cross-Attention (timestep 미적용 — VLM cond 그대로)
            cross_out, _ = self.cross_attn_list[i](self.norm2_list[i](h), c, c)
            h = h + cross_out

            # MLP (AdaLN-Zero)
            h = h + γ2 * self.mlp_list[i](self.norm3_list[i](h) * (1 + α2) + β2)

        return self.output_head(h)                   # (B, H, action_dim)

    # ─────────────────────────────────────────────────────────────────────────
    def get_loss(self, x_1: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Conditional Flow Matching loss.
        x_t = (1-t)*x_0 + t*x_1,  v_target = x_1 - x_0
        """
        B      = x_1.shape[0]
        device = x_1.device
        dtype  = x_1.dtype

        t       = torch.rand(B, device=device, dtype=dtype)          # (B,)
        x_0     = torch.randn_like(x_1)

        t_exp   = t.view(B, 1, 1)
        x_t     = (1 - t_exp) * x_0 + t_exp * x_1
        v_target = x_1 - x_0

        v_pred  = self.forward(x_t, t, cond)
        return F.mse_loss(v_pred, v_target)
