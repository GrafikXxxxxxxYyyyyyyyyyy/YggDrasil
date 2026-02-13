from __future__ import annotations

import torch
from abc import abstractmethod
from typing import Tuple, Optional

from ...core.block.base import AbstractBlock
from ...core.block.registry import register_block


@register_block("position/abstract")
class AbstractPositionEmbedder(AbstractBlock):
    """Позиционные эмбеддинги любой размерности (RoPE, sinusoidal, learned)."""
    
    block_type = "position/abstract"
    
    def _define_slots(self):
        return {}
    
    @abstractmethod
    def __call__(
        self,
        timestep: torch.Tensor,
        shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """timestep → positional embedding подходящей размерности.
        
        shape = (B, C, *spatial) — чтобы правильно broadcast.
        """
        pass