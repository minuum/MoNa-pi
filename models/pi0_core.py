import torch
import torch.nn as nn


class Pi0VLA(nn.Module):
    """
    MoNa-pi Main Model Wrapper — π0 논문 아키텍처

    use_paligemma=True (기본, 권장):
        PaliGemmaBackbone + ActionExpert
        → 실제 π0 논문 구조
        → VLM 전체 토큰 시퀀스를 ActionExpert에 전달

    use_paligemma=False (fallback):
        Pi0Backbone (SigLIP + PerceiverResampler + Gemma) + FlowMatchingHead
        → 이전 구현 (하위 호환)
    """

    def __init__(
        self,
        action_dim: int = 3,
        horizon: int = 10,
        hidden_dim: int = 512,
        use_paligemma: bool = True,
        load_pretrained_paligemma: bool = False,
        use_int8: bool = False,
        use_lora: bool = False,
        lora_r: int = 16,
        use_gemma_expert: bool = False,
        load_lerobot: bool = False,
        vision_model_id: str = "google/siglip-so400m-patch14-384",
        lang_model_id: str = "google/gemma-2b",
        paligemma_id: str = "google/paligemma-3b-pt-224",
        use_bbox_cond: bool = False,
        bbox_only: bool = False,
        bbox_dim: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.use_paligemma = use_paligemma
        # Phase 3 — gray basket bbox(cx,cy,area,valid) grounding 신호 ablation.
        # bbox_only=True면 VLM cond를 아예 안 쓰고 bbox 토큰만 cross-attn에 전달
        # (MoNaVLA ablate_bbox_image_features.py와 동일 철학의 3-way 비교용).
        self.use_bbox_cond = use_bbox_cond or bbox_only
        self.bbox_only = bbox_only

        # ── 1. Backbone ───────────────────────────────────────────────
        if use_paligemma:
            from .backbones.paligemma_backbone import PaliGemmaBackbone
            self.backbone = PaliGemmaBackbone(
                paligemma_id=paligemma_id,
                siglip_id=vision_model_id,
                gemma_id=lang_model_id,
                load_pretrained_paligemma=load_pretrained_paligemma,
                use_int8=use_int8,
                use_lora=use_lora,
                lora_r=lora_r,
                max_text_len=kwargs.get("max_text_len", 48),
            )
        else:
            from .backbones.pi0_backbone import Pi0Backbone
            self.backbone = Pi0Backbone(
                vision_model_id=vision_model_id,
                lang_model_id=lang_model_id,
                **kwargs,
            )

        cond_dim = self.backbone.lang_hidden_size  # 2048

        if self.use_bbox_cond:
            self.bbox_proj = nn.Linear(bbox_dim, cond_dim)

        # ── 2. MoNa-pi Action Expert (SOTA-ready) ────────────────────
        from .heads.mona_action_expert import MoNaActionExpert
        self.action_expert = MoNaActionExpert(
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            use_gemma_expert=use_gemma_expert,
            load_lerobot=load_lerobot,
        )


        # INT8/LoRA 모델은 bfloat16() 재캐스팅 금지
        if not use_int8 and not use_lora:
            self.bfloat16()
        elif use_lora and not use_int8:
            # LoRA: backbone은 LoRA가 이미 bfloat16, action_expert만 캐스팅
            self.action_expert.bfloat16()

    # ─────────────────────────────────────────────────────────────────
    def forward_backbone(self, images: torch.Tensor, instructions, bbox: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            images:       (B, N, C, H, W)
            instructions: List[str] 길이 B
            bbox:         (B, window_size, bbox_dim) gray basket grounding 신호 (Phase 3, 선택)
        Returns:
            cond: (B, T_vlm, 2048)
                use_paligemma=True:  T_vlm = T_vis + L  (전체 VLM 시퀀스)
                use_paligemma=False: T_vlm = 64         (PerceiverResampler latent)
                use_bbox_cond=True:  T_vlm += window_size (bbox 토큰 concat)
                bbox_only=True:      T_vlm = window_size  (VLM cond 자체를 안 씀)
        """
        if self.bbox_only:
            assert bbox is not None, "bbox_only=True인데 bbox가 전달되지 않음"
            return self.bbox_proj(bbox.to(next(self.parameters()).dtype))

        cond = self.backbone(images, instructions)
        if self.use_bbox_cond and bbox is not None:
            bbox_tok = self.bbox_proj(bbox.to(cond.dtype))
            cond = torch.cat([cond, bbox_tok], dim=1)
        return cond

    def compute_loss(
        self,
        images: torch.Tensor,
        instructions,
        actions_gt: torch.Tensor,
        bbox: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        MoNa-pi Flow Matching 학습 손실.
        MoNaActionExpert 내부에서 actions_gt 정규화 후 Loss 계산.
        """
        model_dtype = next(self.parameters()).dtype
        images     = images.to(dtype=model_dtype)
        actions_gt = actions_gt.to(dtype=model_dtype)

        cond = self.forward_backbone(images, instructions, bbox=bbox)
        return self.action_expert.get_loss(actions_gt, cond)

    @torch.no_grad()
    def sample_actions(
        self,
        images: torch.Tensor,
        instructions,
        n_steps: int = 5,
        bbox: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Heun's method ODE solver로 액션 청크 샘플링 및 역정규화.

        Returns:
            x_raw: (B, horizon, action_dim) - 실제 로봇 물리값 (vx, vy, wz)
        """
        model_dtype = next(self.parameters()).dtype
        images = images.to(dtype=model_dtype)
        device = images.device
        dtype  = model_dtype

        cond = self.forward_backbone(images, instructions, bbox=bbox)
        B    = cond.shape[0]

        # 초기 노이즈 (정규화된 공간)
        x_t = torch.randn(
            B, self.action_expert.horizon, self.action_expert.action_dim,
            device=device, dtype=dtype,
        )

        # Heun's Method
        for i in range(n_steps):
            t_curr = i / n_steps
            t_next = (i + 1) / n_steps
            dt     = t_next - t_curr

            t_c = torch.full((B,), t_curr, device=device, dtype=dtype)
            v_t = self.action_expert(x_t, t_c, cond)

            x_next = x_t + v_t * dt

            t_n    = torch.full((B,), t_next, device=device, dtype=dtype)
            v_next = self.action_expert(x_next, t_n, cond)

            x_t = x_t + (v_t + v_next) * 0.5 * dt

        # [Important] 역정규화: 모델 출력(~1.0) -> 로봇 물리값(1.150 등)
        x_raw = self.action_expert.unnormalize(x_t)

        return x_raw
