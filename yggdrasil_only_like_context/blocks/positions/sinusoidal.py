"""Sinusoidal timestep embeddings (classic DDPM)."""
from __future__ import annotations

import torch
import torch.nn as nn
import math
from typing import Tuple
from omegaconf import DictConfig

from ...core.model.position import AbstractPositionEmbedder
from ...core.block.registry import register_block


@register_block("position/sinusoidal")
class SinusoidalEmbedder(AbstractPositionEmbedder):
    """Classic sinusoidal timestep embedding as in DDPM / SD 1.5.
    
    Embeds scalar timestep into a fixed-dimension vector using
    sin/cos at different frequencies.
    """
    
    block_type = "position/sinusoidal"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {})
        self.dim = int(self.config.get("dim", 320))
        self.max_period = float(self.config.get("max_period", 10000.0))
    
    def _forward_impl(
        self,
        timestep: torch.Tensor,
        shape: Tuple[int, ...] = None,
    ) -> torch.Tensor:
        return self._embed(timestep)
    
    def __call__(
        self,
        timestep: torch.Tensor,
        shape: Tuple[int, ...] = None,
    ) -> torch.Tensor:
        return self._embed(timestep)
    
    def _embed(self, timestep: torch.Tensor) -> torch.Tensor:
        """timestep [B] or scalar -> [B, dim]."""
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=timestep.device)
            / half
        )
        
        args = timestep.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        return embedding
