import gc
import os
import torch
import torch.nn as nn
from transformers import AutoModel, GemmaForCausalLM, AutoTokenizer


class PaliGemmaBackbone(nn.Module):
    """
    π0 논문 방식 PaliGemma 백본

    구조 (real π0):
        SigLIP SO400M ──→ multi_modal_projector (Linear 1152→2048)  ─┐
                                                                      ├─→ Gemma-2B encoder → (B, T_vis+L, 2048)
        Tokenizer ──────→ embed_tokens ─────────────────────────────  ─┘

    PaliGemma vs 기존 Pi0Backbone 차이:
        - PerceiverResampler 없음 → Linear projector (PaliGemma 방식)
        - T_vis 토큰 수 고정 없음 (SigLIP 패치 수 그대로 사용)
        - 멀티프레임: 최신 프레임 1개만 사용 (추후 all-frame 확장 가능)

    가중치 전략:
        load_pretrained_paligemma=False (기본):
            로컬 캐시(SigLIP + Gemma)에서 각각 로드,
            multi_modal_projector는 Xavier 랜덤 초기화
        load_pretrained_paligemma=True:
            google/paligemma-3b-pt-224 다운로드 (공동 사전학습 가중치)
            이미지 224px 리사이즈 필요
    """

    def __init__(
        self,
        paligemma_id: str = "google/paligemma-3b-pt-224",
        siglip_id: str = "google/siglip-so400m-patch14-384",
        gemma_id: str = "google/gemma-2b",
        load_pretrained_paligemma: bool = False,
        use_int8: bool = False,
        max_text_len: int = 48,
        **kwargs,
    ):
        super().__init__()
        self.max_text_len = max_text_len
        self.use_int8 = use_int8
        token = os.getenv("HF_TOKEN")

        # INT8: BitsAndBytes 양자화 설정 (MoNaVLA monavla-driving 방식)
        # BF16 6GB → INT8 약 3GB, Serbot2 스왑 환경에서 안정 동작
        bnb_config = None
        if use_int8:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

        if load_pretrained_paligemma:
            # ── Real PaliGemma: 공동 사전학습 가중치 ─────────────────
            from transformers import PaliGemmaForConditionalGeneration
            dtype_str = "INT8" if use_int8 else "BF16"
            print(f"Loading PaliGemma: {paligemma_id} ({dtype_str})...")
            kwargs_pg = dict(token=token, low_cpu_mem_usage=True)
            if use_int8:
                kwargs_pg["quantization_config"] = bnb_config
            else:
                kwargs_pg["torch_dtype"] = torch.bfloat16
            pg = PaliGemmaForConditionalGeneration.from_pretrained(paligemma_id, **kwargs_pg)
            self.vision_tower   = pg.vision_tower
            self.projector      = pg.multi_modal_projector
            # pg.language_model 은 GemmaForCausalLM — 내부 GemmaModel 을 뽑아야 embed_tokens 접근 가능
            lm = pg.language_model
            self.language_model = lm.model if hasattr(lm, "model") else lm
            self.tokenizer = AutoTokenizer.from_pretrained(paligemma_id, token=token)
            del pg; gc.collect()

        else:
            # ── 로컬 캐시: SigLIP + Gemma 별도 로드 ──────────────────
            dtype_str = "INT8" if use_int8 else "BF16"
            print(f"Loading Vision Tower: {siglip_id} ({dtype_str})...")
            kw_vis = dict(token=token, low_cpu_mem_usage=True)
            if use_int8:
                kw_vis["quantization_config"] = bnb_config
            else:
                kw_vis["torch_dtype"] = torch.bfloat16
            siglip = AutoModel.from_pretrained(siglip_id, **kw_vis)
            self.vision_tower = siglip.vision_model
            vision_hidden = siglip.config.vision_config.hidden_size
            del siglip; gc.collect()

            print(f"Loading Language Model: {gemma_id} ({dtype_str})...")
            kw_lm = dict(token=token, low_cpu_mem_usage=True)
            if use_int8:
                kw_lm["quantization_config"] = bnb_config
            else:
                kw_lm["torch_dtype"] = torch.bfloat16
            _full_lm = GemmaForCausalLM.from_pretrained(gemma_id, **kw_lm)
            self.language_model = _full_lm.model
            lang_hidden = _full_lm.config.hidden_size
            del _full_lm; gc.collect()

            self.projector = nn.Linear(vision_hidden, lang_hidden, bias=True)
            nn.init.xavier_uniform_(self.projector.weight)
            nn.init.zeros_(self.projector.bias)

            self.tokenizer = AutoTokenizer.from_pretrained(gemma_id, token=token)

        # ── 공통 ──────────────────────────────────────────────────────
        self.lang_hidden_size = self.language_model.config.hidden_size  # 2048

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # INT8 모델은 bfloat16() 재캐스팅 금지 (이미 양자화됨)
        if not use_int8:
            self.bfloat16()

    # ─────────────────────────────────────────────────────────────────
    def forward(self, images: torch.Tensor, text_input=None) -> torch.Tensor:
        """
        Args:
            images:     (B, N, C, H, W)  — N 프레임 시계열, 최신 프레임만 사용
            text_input: List[str] 길이 B  — 자연어 지시문 (None 허용)
        Returns:
            (B, T_vis + L, 2048)  — VLM 전체 토큰 시퀀스 (ActionExpert에 전달)
            T_vis: SigLIP 패치 수 (384px 기준 ≈729, 224px 기준 256)
            L: 텍스트 토큰 수 (max_text_len 이하)
        """
        B, N, C, H, W = images.shape
        dtype  = next(self.parameters()).dtype  # backbone dtype (BF16)
        device = images.device

        # ── Step 1: 최신 프레임 → SigLIP 비전 인코딩 ─────────────────
        frame = images[:, -1].to(dtype)  # (B, C, H, W)
        vis_out = self.vision_tower(frame)
        visual_feats  = vis_out.last_hidden_state.to(dtype)  # (B, T_vis, 1152)

        # ── Step 2: multi_modal_projector → Gemma 임베딩 공간 ────────
        visual_embeds = self.projector(visual_feats)          # (B, T_vis, 2048)
        T_vis = visual_embeds.shape[1]

        # ── Step 3: 텍스트 처리 ───────────────────────────────────────
        if text_input is not None:
            enc = self.tokenizer(
                text_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_text_len,
            )
            input_ids = enc.input_ids.to(device)
            text_mask = enc.attention_mask.to(device)

            text_embeds = self.language_model.embed_tokens(input_ids).to(dtype)
            # (B, L, 2048)

            combined      = torch.cat([visual_embeds, text_embeds], dim=1)
            # (B, T_vis+L, 2048)
            vis_mask      = torch.ones(B, T_vis, device=device, dtype=text_mask.dtype)
            combined_mask = torch.cat([vis_mask, text_mask], dim=1)
            # (B, T_vis+L)
        else:
            combined      = visual_embeds
            combined_mask = torch.ones(B, T_vis, device=device, dtype=torch.long)

        # ── Step 4: Gemma transformer (인코더 모드, causal mask 없음) ─
        out = self.language_model(
            inputs_embeds=combined,
            attention_mask=combined_mask,
        )
        return out.last_hidden_state  # (B, T_vis+L, 2048)
