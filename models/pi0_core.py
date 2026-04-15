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
        vision_model_id: str = "google/siglip-so400m-patch14-384",
        lang_model_id: str = "google/gemma-2b",
        paligemma_id: str = "google/paligemma-3b-pt-224",
        **kwargs,
    ):
        super().__init__()
        self.use_paligemma = use_paligemma

        # ── 1. Backbone ───────────────────────────────────────────────
        if use_paligemma:
            from .backbones.paligemma_backbone import PaliGemmaBackbone
            self.backbone = PaliGemmaBackbone(
                paligemma_id=paligemma_id,
                siglip_id=vision_model_id,
                gemma_id=lang_model_id,
                load_pretrained_paligemma=load_pretrained_paligemma,
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

        # ── 2. Action Expert (= π0 Action Expert) ────────────────────
        from .heads.action_expert import ActionExpert
        self.action_expert = ActionExpert(
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
        )

        # backward compat: training/train.py, inference/engine.py 등이
        # self.flow_head.horizon / self.flow_head.action_dim 참조
        self.flow_head = self.action_expert._head

        # ── 3. 전체 FP16 (Jetson AGX) ─────────────────────────────────
        self.half()

    # ─────────────────────────────────────────────────────────────────
    def forward_backbone(self, images: torch.Tensor, instructions) -> torch.Tensor:
        """
        Args:
            images:       (B, N, C, H, W)
            instructions: List[str] 길이 B
        Returns:
            cond: (B, T_vlm, 2048)
                use_paligemma=True:  T_vlm = T_vis + L  (전체 VLM 시퀀스)
                use_paligemma=False: T_vlm = 64         (PerceiverResampler latent)
        """
        return self.backbone(images, instructions)

    def compute_loss(
        self,
        images: torch.Tensor,
        instructions,
        actions_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flow Matching 학습 손실.
        fp32 DataLoader 텐서도 모델 dtype(fp16)으로 자동 캐스팅.
        """
        model_dtype = next(self.parameters()).dtype
        images     = images.to(dtype=model_dtype)
        actions_gt = actions_gt.to(dtype=model_dtype)

        cond = self.forward_backbone(images, instructions)
        return self.action_expert.get_loss(actions_gt, cond)

    @torch.no_grad()
    def sample_actions(
        self,
        images: torch.Tensor,
        instructions,
        n_steps: int = 5,
    ) -> torch.Tensor:
        """
        Heun's method ODE solver로 액션 청크 샘플링.

        Returns:
            x_t: (B, horizon, action_dim)
        """
        model_dtype = next(self.parameters()).dtype
        images = images.to(dtype=model_dtype)
        device = images.device
        dtype  = model_dtype

        cond = self.forward_backbone(images, instructions)
        B    = cond.shape[0]

        # 초기 노이즈
        x_t = torch.randn(
            B, self.flow_head.horizon, self.flow_head.action_dim,
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

        return x_t
