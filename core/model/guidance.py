from __future__ import annotations

import torch
from abc import abstractmethod
from typing import Optional, Dict, Any

from ...core.block.base import AbstractBlock
from ...core.block.registry import register_block

from .modular import ModularDiffusionModel


@register_block("guidance/abstract")
class AbstractGuidance(AbstractBlock):
    """Методы guidance (CFG, PAG, FreeU, custom)."""
    
    block_type = "guidance/abstract"
    
    def _define_slots(self):
        return {}
    
    @abstractmethod
    def __call__(
        self,
        model_output: torch.Tensor,
        condition: Optional[Dict[str, Any]] = None,
        model: Optional[ModularDiffusionModel] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """Применить guidance к выходу backbone.
        
        Возвращает модифицированный model_output.
        """
        pass