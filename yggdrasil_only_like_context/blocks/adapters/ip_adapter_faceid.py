"""IP-Adapter FaceID — face embeddings projection for identity-preserving generation.

Lego block: conditioner/faceid -> adapter/ip_adapter_faceid -> image_prompt_embeds.
"""
from __future__ import annotations

import logging
import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Any, Dict

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import InputPort, OutputPort
from .base import AbstractAdapter
from .ip_adapter import ImageProjection

logger = logging.getLogger(__name__)


@register_block("adapter/ip_adapter_faceid")
class IPAdapterFaceID(AbstractAdapter):
    """IP-Adapter FaceID — projects 512-dim face embeddings to cross-attention.

    Lego: conditioner/faceid (512) -> image_features -> image_prompt_embeds.
    """

    block_type = "adapter/ip_adapter_faceid"

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.scale = config.get("scale", 1.0)
        face_embed_dim = config.get("face_embed_dim", 512)
        self.cross_attention_dim = config.get("cross_attention_dim", 768)
        num_tokens = config.get("num_tokens", 4)
        self.image_proj = ImageProjection(
            image_embed_dim=config.get("face_embed_dim", 512),
            cross_attention_dim=self.cross_attention_dim,
            num_tokens=num_tokens,
        )

    @classmethod
    def declare_io(cls) -> dict:
        return {
            "image_features": InputPort("image_features", optional=True,
                description="Face embeddings (N, 512) from conditioner/faceid."),
            "image_embeds": InputPort("image_embeds", optional=True,
                description="Pre-computed projected tokens (bypass)."),
            "image_prompt_embeds": OutputPort("image_prompt_embeds",
                description="Projected tokens for cross-attention"),
        }

    def process(self, **kw) -> Dict[str, Any]:
        embeds = kw.get("image_embeds")
        if embeds is not None:
            return {"image_prompt_embeds": embeds}
        feats = kw.get("image_features")
        if feats is None:
            return {"image_prompt_embeds": None}
        if feats.dim() == 3:
            feats = feats.squeeze(1)
        device = next(self.image_proj.parameters()).device
        dtype = next(self.image_proj.parameters()).dtype
        feats = feats.to(device=device, dtype=dtype)
        n = feats.shape[0]
        proj = self.image_proj(feats)
        if n > 1:
            proj = proj.reshape(1, -1, proj.shape[-1])
        return {"image_prompt_embeds": proj}

    def inject_into(self, target):
        pass

    def apply(self, output: torch.Tensor, context=None) -> torch.Tensor:
        return output

    def load_weights(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "image_proj" in state:
            self.image_proj.load_state_dict(state["image_proj"], strict=False)
        logger.info(f"Loaded IP-Adapter FaceID weights from {path}")

    def save_weights(self, path: str):
        torch.save({"image_proj": self.image_proj.state_dict()}, path)
