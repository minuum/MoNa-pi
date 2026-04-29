import torch
import torch.nn as nn
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


class PaliGemmaBackbone(nn.Module):
    """
    SigLIP + Gemma-2B backbone (π0 방식).

    forward()는 이미지 윈도우와 언어 명령어를 받아
    Gemma의 마지막 히든 스테이트 전체 시퀀스를 반환한다.
    Action Expert가 이 토큰들을 cross-attention 조건으로 사용한다.

    Args:
        model_id: HuggingFace model ID 또는 로컬 캐시 경로
        load_pretrained: False면 랜덤 초기화 (테스트용)
        freeze_vision: SigLIP 비전 인코더 가중치 동결 여부
        freeze_language: Gemma LM 가중치 동결 여부
        dtype: 모델 dtype (torch.float16 권장)
    """
    MODEL_ID = "google/paligemma-3b-pt-224"
    IMG_SIZE = 224

    def __init__(
        self,
        model_id: str = None,
        load_pretrained: bool = True,
        freeze_vision: bool = True,
        freeze_language: bool = False,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.dtype = dtype
        _id = model_id or self.MODEL_ID

        if load_pretrained:
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                _id, torch_dtype=dtype
            )
            self.processor = AutoProcessor.from_pretrained(_id)
        else:
            # 완전 오프라인 mock — config를 직접 구성해 네트워크 접근 없이 초기화
            self.model = self._build_mock_model(dtype)
            self.processor = None  # mock에서는 processor 직접 사용 안 함

        if not load_pretrained:
            self._gemma_hidden = 2048  # PaliGemma-3B Gemma2 hidden size
        else:
            # transformers 5.x: 서브모듈은 model.model 아래에 있음
            inner = self.model.model
            if freeze_vision:
                for p in inner.vision_tower.parameters():
                    p.requires_grad_(False)
            if freeze_language:
                for p in inner.language_model.parameters():
                    p.requires_grad_(False)
            self._gemma_hidden = self.model.config.text_config.hidden_size

        # Gemma hidden size → backbone_out_dim (1024 고정)
        self.proj = nn.Linear(self._gemma_hidden, 1024)

    @staticmethod
    def _build_mock_model(dtype):
        """네트워크 없이 PaliGemma 구조를 랜덤 초기화해 반환."""
        from transformers import (
            PaliGemmaConfig, SiglipVisionConfig, Gemma2Config,
        )
        vision_cfg = SiglipVisionConfig(
            hidden_size=1152,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
            image_size=224,
            patch_size=14,
        )
        text_cfg = Gemma2Config(
            hidden_size=2048,
            intermediate_size=16384,
            num_hidden_layers=18,
            num_attention_heads=8,
            num_key_value_heads=1,
            vocab_size=257216,
        )
        cfg = PaliGemmaConfig(
            vision_config=vision_cfg,
            text_config=text_cfg,
            projection_dim=2048,
        )
        return PaliGemmaForConditionalGeneration(cfg).to(dtype)

    @property
    def out_dim(self) -> int:
        return 1024

    def forward(
        self,
        images: torch.Tensor,       # (B, W, 3, H, H)  W = window_size
        instructions: list[str],    # len = B
    ) -> torch.Tensor:
        """
        Returns:
            cond: (B, n_tokens, 1024)  — Action Expert 조건 시퀀스
        """
        B, W, C, H, _ = images.shape
        device = images.device
        imgs_flat = images.view(B * W, C, H, H)  # (B*W, 3, H, H)

        # 이미지 패치 수 계산 (mock/real 공통)
        n_patches = (self.model.config.vision_config.image_size //
                     self.model.config.vision_config.patch_size) ** 2
        img_token_id = getattr(self.model.config, "image_token_index", 257152)
        BW = B * W

        if self.processor is not None:
            # 실제 가중치 모드 — tokenizer로 텍스트만 토크나이징, pixel_values 직접 전달
            repeated_instructions = [inst for inst in instructions for _ in range(W)]
            tok = self.processor.tokenizer(
                repeated_instructions,
                return_tensors="pt",
                padding=True,
            ).to(device)
            # input_ids 앞에 이미지 토큰 n_patches개 prepend
            img_ids = torch.full((BW, n_patches), img_token_id, dtype=torch.long, device=device)
            input_ids = torch.cat([img_ids, tok.input_ids], dim=1)
            attn_mask = torch.cat([
                torch.ones((BW, n_patches), dtype=torch.long, device=device),
                tok.attention_mask,
            ], dim=1)
            token_type_ids = torch.cat([
                torch.zeros((BW, n_patches), dtype=torch.long, device=device),
                torch.ones_like(tok.attention_mask),
            ], dim=1)
            inputs = {
                "pixel_values": imgs_flat.to(self.dtype),
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "token_type_ids": token_type_ids,
            }
        else:
            # mock 모드 — 더미 토큰으로 오프라인 구조 검증
            n_text = 8
            img_ids  = torch.full((BW, n_patches), img_token_id, dtype=torch.long, device=device)
            text_ids = torch.ones((BW, n_text), dtype=torch.long, device=device)
            input_ids = torch.cat([img_ids, text_ids], dim=1)
            token_type_ids = torch.cat([
                torch.zeros((BW, n_patches), dtype=torch.long, device=device),
                torch.ones((BW, n_text),    dtype=torch.long, device=device),
            ], dim=1)
            inputs = {
                "pixel_values": imgs_flat.to(self.dtype),
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "token_type_ids": token_type_ids,
            }

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        # 마지막 레이어 히든 스테이트: (B*W, seq_len, gemma_hidden)
        hidden = outputs.hidden_states[-1]
        hidden = self.proj(hidden.to(torch.float32))  # (B*W, seq_len, 1024)

        # window 평균 → (B, seq_len, 1024)
        _, seq_len, d = hidden.shape
        hidden = hidden.view(B, W, seq_len, d).mean(dim=1)

        return hidden
