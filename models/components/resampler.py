import torch
import torch.nn as nn
from einops import rearrange, repeat

class PerceiverAttention(nn.Module):
    def __init__(self, dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.half()

    def forward(self, x, context):
        h = self.heads
        dtype = x.dtype
        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), kv)
        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        scale = torch.tensor(self.scale, device=x.device, dtype=dtype)
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    """
    SigLIP (Vision) -> Gemma (Language) 브릿지 모듈
    """
    def __init__(
        self,
        dim=1024,             # Gemma embedding dim (approx)
        depth=6,
        heads=8,
        dim_head=64,
        num_latents=64,       # 고정된 시각 토큰 개수
        num_media_embeds=4,   # 지원할 최대 카메라 개수
        context_dim=1152      # SigLIP output dim
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, context_dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim, context_dim, heads=heads, dim_head=dim_head),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                )
            ]))
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: (B, N_images, N_tokens, context_dim)
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)
        
        b, n, t, d = x.shape
        dtype = x.dtype
        
        # Add media position embeddings (camera-wise awareness)
        pos_emb = repeat(self.media_pos_emb[:n], "n 1 d -> b n t d", b=b, t=t).to(dtype)
        x = x + pos_emb
        
        # Flatten cameras and tokens
        x = rearrange(x, "b n t d -> b (n t) d")
        
        latents = repeat(self.latents, "n d -> b n d", b=b).to(dtype)
        
        for attn, ff in self.layers:
            latents = attn(latents, x) + latents
            latents = ff(latents) + latents
            
        return self.norm(latents) # (B, num_latents, dim)
