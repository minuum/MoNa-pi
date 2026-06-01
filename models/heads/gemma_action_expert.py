"""
GemmaActionExpert — π0 정통 방식 Action Expert

π0 원본 아키텍처:
    VLM 토큰 + action 토큰을 concatenate하여 동일한 Gemma transformer에 통과
    → VLM ↔ action 간 통합 self-attention (cross-attention 아님)

vs 기존 FlowMatchingHead:
    FlowMatchingHead: action 토큰이 VLM features를 cross-attention으로만 참조
    GemmaActionExpert: VLM 토큰과 action 토큰이 같은 공간에서 함께 self-attention

구조:
    input_dim (2048) ──→ vlm_proj (Linear) ──→ (B, T_vlm, 1024)  ─┐
                                                                     ├→ concat → GemmaLayers → action[:, T_vlm:] → out_proj
    action_dim (3) ────→ action_in (Linear) ──→ (B, H, 1024)  ────┘
    timestep ──────────→ TimestepEmbedder ──→ (B, 1024) → add to action tokens

초기화 옵션:
    load_lerobot=True → checkpoints/lerobot_gemma_expert/ 가중치로 초기화
    (lerobot/pi0_old의 로봇 데이터 적응 Gemma expert)
"""

from __future__ import annotations
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaModel

from models.heads.flow_head import TimestepEmbedder


class GemmaActionExpert(nn.Module):
    """
    Gemma transformer 기반 Action Expert.

    Args:
        action_dim:     로봇 액션 차원 (3: vx, vy, wz)
        horizon:        예측 청크 길이 (10)
        cond_dim:       VLM backbone 출력 차원 (2048)
        hidden_size:    Gemma hidden size (1024, lerobot 동일)
        n_layers:       Gemma 레이어 수 (기본 4, lerobot은 18)
        load_lerobot:   lerobot gemma_expert 가중치로 초기화
    """

    # lerobot/pi0_old gemma_expert 스펙 (18-layer, hidden=1024, MQA)
    GEMMA_CONFIG = dict(
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=8,
        num_key_value_heads=1,  # MQA
        head_dim=256,
        hidden_act="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        max_position_embeddings=600,  # T_vis+L + horizon 여유
        vocab_size=257152,            # Gemma vocab (embed 로딩용)
        pad_token_id=0,
    )

    def __init__(
        self,
        action_dim: int = 3,
        horizon: int = 10,
        cond_dim: int = 2048,
        hidden_size: int = 1024,
        n_layers: int = 4,
        load_lerobot: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon    = horizon
        self.hidden_size = hidden_size

        # Gemma transformer (n_layers만 사용)
        cfg = GemmaConfig(**{**self.GEMMA_CONFIG, "hidden_size": hidden_size,
                             "num_hidden_layers": n_layers})
        self.gemma = GemmaModel(cfg)

        # 입출력 프로젝션
        self.vlm_proj   = nn.Linear(cond_dim,   hidden_size, bias=False)
        self.action_in  = nn.Linear(action_dim, hidden_size, bias=False)
        self.action_out = nn.Linear(hidden_size, action_dim, bias=False)

        # action 토큰 위치 임베딩 (VLM과 구분)
        self.action_pos_emb = nn.Parameter(torch.zeros(1, horizon, hidden_size))

        # 타임스텝 임베더
        self.timestep_emb = TimestepEmbedder(hidden_size)

        # 정규화 스케일 (Gemma 초기화 관례)
        nn.init.normal_(self.action_in.weight,  std=0.02)
        nn.init.normal_(self.action_out.weight, std=0.02)
        nn.init.normal_(self.vlm_proj.weight,   std=0.02)

        if load_lerobot:
            self._load_lerobot_expert(n_layers)

    def _load_lerobot_expert(self, n_layers: int):
        """lerobot/pi0_old gemma_expert 첫 n_layers 레이어로 초기화."""
        ckpt = Path("checkpoints/lerobot_gemma_expert/model.safetensors")
        if not ckpt.exists():
            print(f"[GemmaExpert] lerobot 가중치 없음: {ckpt}, 랜덤 초기화 유지")
            return
        from safetensors.torch import load_file
        state = load_file(str(ckpt), device="cpu")
        model_dict = self.state_dict()
        # n_layers 범위의 레이어만 로드
        matched = {k: v for k, v in state.items()
                   if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        self.load_state_dict(model_dict)
        print(f"[GemmaExpert] lerobot 가중치 로드: {len(matched)}/{len(model_dict)}개 매칭")

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t:  (B, horizon, action_dim)  노이즈 액션
            t:    (B,)                       타임스텝
            cond: (B, T_vlm, cond_dim)       VLM 조건
        Returns:
            velocity: (B, horizon, action_dim)
        """
        B, H, _ = x_t.shape
        T_vlm = cond.shape[1]
        dtype  = next(self.parameters()).dtype
        device = x_t.device

        # 1. VLM 토큰 프로젝션
        vlm_tokens = self.vlm_proj(cond.to(dtype))   # (B, T_vlm, hidden)

        # 2. action 토큰 프로젝션
        act_tokens = self.action_in(x_t.to(dtype))   # (B, H, hidden)

        # 3. 타임스텝 임베딩 → action 토큰에 추가
        t_emb = self.timestep_emb(t.to(dtype))       # (B, hidden)
        act_tokens = act_tokens + t_emb.unsqueeze(1) + self.action_pos_emb

        # 4. VLM 토큰 + action 토큰 concatenate
        combined = torch.cat([vlm_tokens, act_tokens], dim=1)  # (B, T_vlm+H, hidden)

        # 5. position_ids
        position_ids = torch.arange(T_vlm + H, device=device).unsqueeze(0).expand(B, -1)

        # 6. Gemma transformer (통합 self-attention)
        out = self.gemma(inputs_embeds=combined, position_ids=position_ids)
        hidden = out.last_hidden_state   # (B, T_vlm+H, hidden)

        # 7. action 위치 추출 → 출력
        action_hidden = hidden[:, T_vlm:, :]          # (B, H, hidden)
        return self.action_out(action_hidden)          # (B, H, action_dim)

    def get_loss(self, x_1: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Conditional Flow Matching loss (FlowMatchingHead와 동일)."""
        B = x_1.shape[0]
        device = x_1.device
        dtype  = next(self.parameters()).dtype

        t   = torch.rand(B, device=device, dtype=dtype)
        x_0 = torch.randn_like(x_1)
        x_t = (1 - t[:, None, None]) * x_0 + t[:, None, None] * x_1
        v_target = x_1 - x_0

        v_pred = self.forward(x_t, t, cond)
        return F.mse_loss(v_pred, v_target.to(dtype))
