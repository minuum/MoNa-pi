import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, GemmaForCausalLM, AutoTokenizer
from ..components.resampler import PerceiverResampler


class Pi0Backbone(nn.Module):
    """
    π0 스타일 Vision-Language Backbone

    구조:
        SigLIP SO400M  ──→  PerceiverResampler (64 latents)  ─┐
                                                               ├─→ Gemma-2B joint forward ──→ (B, 64, 2048)
        AutoTokenizer  ──→  embed_tokens  ──────────────────  ─┘

    π0 논문 방식:
        PaliGemma (SigLIP + Gemma)가 시각·언어 토큰을 공동 어텐션으로 처리.
        [visual_latents(64) | text_tokens(L)] 을 Gemma transformer에 통과시켜
        언어 정보가 반영된 시각 conditioning 생성.

        ← 이전 구현의 문제: text_input이 완전히 무시되어 언어 지시가
          FlowMatchingHead conditioning에 반영되지 않았음.
    """

    def __init__(
        self,
        vision_model_id: str = "google/siglip-so400m-patch14-384",
        lang_model_id: str = "google/gemma-2b",
        num_latents: int = 64,
        max_text_len: int = 48,
        full_seq_cond: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.max_text_len = max_text_len
        self.full_seq_cond = full_seq_cond

        token = os.getenv("HF_TOKEN")

        # ── 1. Vision Encoder: SigLIP SO400M ──────────────────────
        print(f"Loading Vision Encoder: {vision_model_id} (FP16)...")
        self.vision_encoder = AutoModel.from_pretrained(
            vision_model_id,
            token=token,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.vision_processor = AutoProcessor.from_pretrained(
            vision_model_id, token=token
        )
        vision_hidden = self.vision_encoder.config.vision_config.hidden_size  # 1152

        # ── 2. Language Model: Gemma-2B ───────────────────────────
        print(f"Loading Language Model: {lang_model_id} (FP16)...")
        self.lang_model = GemmaForCausalLM.from_pretrained(
            lang_model_id,
            token=token,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.lang_hidden_size = self.lang_model.config.hidden_size  # 2048

        # 텍스트 토크나이저 (pad token 보장)
        self.tokenizer = AutoTokenizer.from_pretrained(lang_model_id, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── 3. PerceiverResampler: SigLIP 1152D → Gemma 2048D ────
        self.resampler = PerceiverResampler(
            dim=self.lang_hidden_size,   # 2048
            context_dim=vision_hidden,   # 1152
            num_latents=num_latents,     # 64
        )

        # 전체 FP16 변환 (Jetson AGX 메모리 최적화)
        self.half()

    # ─────────────────────────────────────────────────────────────
    def forward(self, images: torch.Tensor, text_input=None) -> torch.Tensor:
        """
        Args:
            images:     (B, N_cam, C, H, W)   FP16, CLIP 정규화됨
            text_input: List[str] 길이 B        — 자연어 지시문
        Returns:
            full_seq_cond=False: (B, 64, 2048)     64 latent만 (기본)
            full_seq_cond=True:  (B, 64+L, 2048)   전체 VLM 시퀀스 (π0 Action Expert 방식)
        """
        B, N, C, H, W = images.shape
        dtype  = images.dtype
        device = images.device

        # ── Step 1: SigLIP 시각 특징 추출 ─────────────────────────
        vis_out = self.vision_encoder.vision_model(
            images.view(B * N, C, H, W)
        )
        vision_feats = vis_out.last_hidden_state.to(dtype)          # (B*N, T, 1152)
        vision_feats = vision_feats.view(B, N, -1, vision_feats.shape[-1])

        # ── Step 2: PerceiverResampler → 64 고정 시각 토큰 ────────
        visual_latents = self.resampler(vision_feats)               # (B, 64, 2048)

        # ── Step 3: Gemma-2B로 시각·언어 공동 어텐션 ─────────────
        if text_input is not None:
            # 텍스트 토크나이저 (배치 패딩 포함)
            enc = self.tokenizer(
                text_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_text_len,
            )
            input_ids = enc.input_ids.to(device)          # (B, L)
            text_mask = enc.attention_mask.to(device)      # (B, L)

            # Gemma embed_tokens: 토크나이저 출력 → 임베딩 공간
            text_embeds = self.lang_model.model.embed_tokens(input_ids).to(dtype)
            # (B, L, 2048)

            # [시각 64토큰 | 텍스트 L토큰] 연결
            combined_embeds = torch.cat([visual_latents, text_embeds], dim=1)
            # (B, 64+L, 2048)

            visual_mask = torch.ones(
                B, self.num_latents, device=device, dtype=text_mask.dtype
            )
            combined_mask = torch.cat([visual_mask, text_mask], dim=1)
            # (B, 64+L)

            # Gemma transformer 전체를 인코더 모드로 실행
            # (causal mask 없음 — 시각과 언어 토큰이 서로 어텐션 가능)
            gemma_out = self.lang_model.model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
            )
            hidden = gemma_out.last_hidden_state  # (B, 64+L, 2048)
            if self.full_seq_cond:
                # π0 Action Expert 방식: 전체 VLM 시퀀스를 FlowHead에 전달
                cond = hidden                           # (B, 64+L, 2048)
            else:
                # 기본: 언어 grounded 시각 latent 슬라이스만 반환
                cond = hidden[:, : self.num_latents]   # (B, 64, 2048)
        else:
            # 언어 없을 때: 시각 latent만 반환 (순수 시각 conditioning)
            cond = visual_latents                      # (B, 64, 2048)

        return cond
