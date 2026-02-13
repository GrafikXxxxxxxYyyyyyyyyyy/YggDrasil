import torch
from abc import abstractmethod
from typing import Optional, Tuple

from ....core.block.base import AbstractBlock
from ....core.block.registry import register_block


@register_block("noise/sampler/abstract")
class NoiseSampler(AbstractBlock):
    """Генератор начального шума (Gaussian, LowDiscrepancy, Perlin и т.д.)."""
    
    block_type = "noise/sampler/abstract"
    
    def _define_slots(self):
        return {}
    
    @abstractmethod
    def sample(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Возвращает чистый шум."""
        pass