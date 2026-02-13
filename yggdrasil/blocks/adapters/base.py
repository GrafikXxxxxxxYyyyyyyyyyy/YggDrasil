from __future__ import annotations

import torch 

from abc import abstractmethod
from typing import Dict, Any, Optional
from omegaconf import DictConfig

from ...core.block.base import AbstractBlock
from ...core.block.registry import register_block
from ...core.model.backbone import AbstractBackbone


@register_block("adapter/abstract")
class AbstractAdapter(AbstractBlock):
    """Базовый адаптер — Lego-кирпичик для модификации модели."""
    
    block_type = "adapter/abstract"
    
    def _define_slots(self):
        return {}
    
    @abstractmethod
    def inject_into(self, target: AbstractBackbone | ModularDiffusionModel):
        """Инжектирует себя в целевую модель/backbone."""
        pass
    
    @abstractmethod
    def apply(self, output: torch.Tensor, context: Any = None) -> torch.Tensor:
        """Применяется на каждом forward (если нужно)."""
        return output