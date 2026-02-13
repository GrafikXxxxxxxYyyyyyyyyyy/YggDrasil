# yggdrasil/core/model/guidance.py
from __future__ import annotations

import torch
from abc import abstractmethod
from typing import Optional, Dict, Any
from omegaconf import DictConfig

from yggdrasil.core.block.base import AbstractBlock
from yggdrasil.core.block.registry import register_block


@register_block("guidance/abstract")
class AbstractGuidance(AbstractBlock):
    """Методы guidance (CFG, PAG, FreeU, custom)."""
    
    block_type = "guidance/abstract"
    
    def _define_slots(self):
        return {}
    
    def _forward_impl(self, *args, **kwargs) -> torch.Tensor:
        """Требуется AbstractBlock; делегирует в __call__."""
        return self(*args, **kwargs)

    @abstractmethod
    def __call__(
        self,
        model_output: torch.Tensor,
        condition: Optional[Dict[str, Any]] = None,
        model: Optional["ModularDiffusionModel"] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """Применить guidance к выходу backbone."""
        pass