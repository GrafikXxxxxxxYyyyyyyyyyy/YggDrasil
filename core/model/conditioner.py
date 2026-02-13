from __future__ import annotations

import torch
from abc import abstractmethod
from typing import Dict, Any
from omegaconf import DictConfig

from ...core.block.base import AbstractBlock
from ...core.block.registry import register_block


@register_block("conditioner/abstract")
class AbstractConditioner(AbstractBlock):
    """Обработка условий (текст, ControlNet, IP-Adapter, CLAP...)."""
    
    block_type = "conditioner/abstract"
    
    def _define_slots(self):
        return {}
    
    @abstractmethod
    def __call__(self, condition: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """condition → dict эмбеддингов.
        
        Пример:
            {"text": "cat", "image": img_tensor} → {"text_emb": tensor, "image_emb": tensor}
        """
        pass