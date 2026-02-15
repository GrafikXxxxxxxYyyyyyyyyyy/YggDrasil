from __future__ import annotations

import torch
from abc import abstractmethod
from typing import Tuple, Optional

from ...core.block.base import AbstractBaseBlock
from ...core.block.registry import register_block
from ...core.block.port import Port, InputPort, OutputPort, TensorSpec


@register_block("position/abstract")
class AbstractPositionEmbedder(AbstractBaseBlock):
    """Позиционные эмбеддинги любой размерности (RoPE, sinusoidal, learned)."""
    
    block_type = "position/abstract"
    
    @classmethod
    def declare_io(cls) -> dict:
        return {
            "timestep": InputPort("timestep", data_type="tensor", description="Timestep"),
            "shape": InputPort("shape", data_type="any", optional=True, description="Spatial shape hint"),
            "embedding": OutputPort("embedding", spec=TensorSpec(space="embedding"), description="Position embedding"),
        }
    
    def process(self, **port_inputs) -> dict:
        timestep = port_inputs.get("timestep")
        shape = port_inputs.get("shape")
        emb = self(timestep, shape)
        return {"embedding": emb, "output": emb}
    
    def _define_slots(self):
        return {}

    def _forward_impl(self, *args, **kwargs) -> torch.Tensor:
        """Требуется AbstractBaseBlock; делегирует в __call__(timestep, shape)."""
        return self(*args, **kwargs)
    
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