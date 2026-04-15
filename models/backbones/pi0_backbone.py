import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, GemmaForCausalLM
from ..components.resampler import PerceiverResampler

class Pi0Backbone(nn.Module):
    """
    Original pi0 Backbone Configuration:
    - Vision: SigLIP (SO400M)
    - Language: Gemma-2B
    - Bridge: Perceiver Resampler
    """
    def __init__(
        self,
        vision_model_id="google/siglip-so400m-patch14-384",
        lang_model_id="google/gemma-2b",
        num_latents=64,
        **kwargs
    ):
        super().__init__()
        
        # 1. Vision Encoder (SigLIP)
        token = os.getenv("HF_TOKEN")
        print(f"Loading Vision Encoder: {vision_model_id} (FP16)...")
        self.vision_encoder = AutoModel.from_pretrained(
            vision_model_id, 
            token=token,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.vision_processor = AutoProcessor.from_pretrained(vision_model_id, token=token)
        
        # 2. Language Model (Gemma)
        print(f"Loading Language Model: {lang_model_id} (FP16)...")
        self.lang_model = GemmaForCausalLM.from_pretrained(
            lang_model_id, 
            token=token,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.lang_hidden_size = self.lang_model.config.hidden_size
        
        # 3. Perceiver Resampler (Bridge)
        self.resampler = PerceiverResampler(
            dim=self.lang_hidden_size,
            context_dim=self.vision_encoder.config.vision_config.hidden_size,
            num_latents=num_latents
        )

    def forward(self, images, text_input=None):
        """
        images: (B, N_cam, C, H, W)
        """
        B, N, C, H, W = images.shape
        images = images.view(B * N, C, H, W)
        
        # 1. Extract Vision Features
        # SigLIP returns (B*N, SeqLen, Hidden)
        vision_outputs = self.vision_encoder.vision_model(images)
        vision_features = vision_outputs.last_hidden_state # (B*N, T, D)
        
        # Ensure dtype consistency (SigLIP sometimes outputs float32 even if loaded in fp16)
        vision_features = vision_features.to(images.dtype)
        
        # 2. Reshape for Resampler
        vision_features = vision_features.view(B, N, -1, vision_features.shape[-1])
        
        # 3. Resample to Fixed Visual Tokens
        visual_latents = self.resampler(vision_features) # (B, 64, 2048)
        
        # 4. Integrate Text and Vision (Simple Concat for now)
        # In π0, they often use vis-tokens as prefix tokens for Gemma
        # Here we just return the fused representation for the Flow Head
        return visual_latents
