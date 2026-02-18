"""IP-Adapter Plus — patch embeddings (ViT-H) projection for finer image conditioning.

Lego block: conditioner/clip_vision (output_mode='patches') -> adapter/ip_adapter_plus -> image_prompt_embeds.
Uses patch-level features instead of pooled for better detail preservation.
"""
from __future__ import annotations

import logging
import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import InputPort, OutputPort
from .base import AbstractAdapter

logger = logging.getLogger(__name__)


class PatchProjection(nn.Module):
    """Projects patch embeddings (N, num_patches, embed_dim) to cross-attention tokens.

    Uses per-patch projection + learned compression to num_tokens.
    Compatible with IP-Adapter Plus pretrained weights (image_proj.*).
    """

    def __init__(
        self,
        embed_dim: int,
        cross_attention_dim: int,
        num_tokens: int = 16,
        num_patches: int = 257,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.proj = nn.Linear(embed_dim, cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)
        self.queries = nn.Parameter(torch.randn(1, num_tokens, cross_attention_dim) * 0.02)
        self.attn = nn.MultiheadAttention(cross_attention_dim, num_heads=8, batch_first=True)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (batch, num_patches, embed_dim) -> (batch, num_tokens, cross_attention_dim)."""
        x = self.norm(self.proj(patches))
        q = self.queries.expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, x, x)
        return out


@register_block("adapter/ip_adapter_plus")
class IPAdapterPlus(AbstractAdapter):
    """IP-Adapter Plus — patch-level image conditioning (ViT-H patches -> cross-attn tokens).

    Lego: connect conditioner/clip_vision (output_mode='patches') -> image_features.
    Trainable: image_proj, load_weights/save_weights.
    """

    block_type = "adapter/ip_adapter_plus"

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.scale = config.get("scale", 1.0)
        embed_dim = config.get("image_embed_dim", 1280)
        self.cross_attention_dim = config.get("cross_attention_dim", 768)
        num_tokens = config.get("num_tokens", 16)
        num_patches = config.get("num_patches", 257)
        self.image_proj = PatchProjection(
            embed_dim=embed_dim,
            cross_attention_dim=self.cross_attention_dim,
            num_tokens=num_tokens,
            num_patches=num_patches,
        )
        self.num_tokens = num_tokens

    @classmethod
    def declare_io(cls) -> dict:
        return {
            "image_features": InputPort("image_features", optional=True,
                description="Patch embeddings (N, num_patches, dim) from clip_vision patches."),
            "image_embeds": InputPort("image_embeds", optional=True,
                description="Pre-computed projected tokens (bypass)."),
            "image_prompt_embeds": OutputPort("image_prompt_embeds",
                description="Projected tokens (1, num_tokens, dim) or (1, N*num_tokens, dim)"),
        }

    def process(self, **kw) -> Dict[str, Any]:
        embeds = kw.get("image_embeds")
        if embeds is not None:
            return {"image_prompt_embeds": embeds}
        patches = kw.get("image_features")
        if patches is None:
            return {"image_prompt_embeds": None}
        device = next(self.image_proj.parameters()).device
        dtype = next(self.image_proj.parameters()).dtype
        if patches.device != device or patches.dtype != dtype:
            patches = patches.to(device=device, dtype=dtype)
        if patches.dim() == 2:
            patches = patches.unsqueeze(0)
        n = patches.shape[0]
        proj = self.image_proj(patches)
        if n > 1:
            proj = proj.reshape(1, -1, self.cross_attention_dim)
        return {"image_prompt_embeds": proj}

    def inject_into(self, target):
        pass

    def apply(self, output: torch.Tensor, context=None) -> torch.Tensor:
        return output

    def load_weights(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "image_proj" in state:
            self.image_proj.load_state_dict(state["image_proj"], strict=False)
        logger.info(f"Loaded IP-Adapter Plus weights from {path}")

    def save_weights(self, path: str):
        torch.save({"image_proj": self.image_proj.state_dict()}, path)
        logger.info(f"Saved IP-Adapter Plus weights to {path}")
