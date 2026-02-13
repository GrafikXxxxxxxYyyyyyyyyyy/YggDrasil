from __future__ import annotations

import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Optional, Any, Dict
from omegaconf import DictConfig

from ...core.block.base import AbstractBlock
from ...core.block.registry import register_block
from ...core.block.slot import Slot


@register_block("backbone/abstract")
class AbstractBackbone(AbstractBlock, nn.Module):
    """Абстрактный backbone (UNet, DiT, Transformer, GNN и т.д.).
    
    Любой конкретный backbone (в blocks/backbones/) должен наследоваться отсюда.
    """
    
    block_type = "backbone/abstract"
    
    def _define_slots(self) -> Dict[str, Slot]:
        """Backbone может иметь свои слоты (например, для LoRA внутри)."""
        return {
            "adapters": Slot(
                name="adapters",
                accepts=AbstractBlock,  # LoRA, DoRA и т.д.
                multiple=True,
                optional=True
            )
        }
    
    @abstractmethod
    def _forward_impl(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        position_embedding: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """Основной forward backbone.
        
        Args:
            x: latents [B, C, *spatial_dims]
            timestep: [B] или [B, 1]
            condition: dict эмбеддингов (text, control, ip и т.д.)
            position_embedding: RoPE / sinusoidal / learned
        """
        pass
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Пробрасываем в _forward_impl + применяем адаптеры."""
        output = super().forward(*args, **kwargs)  # вызовет _forward_impl с хуками
        
        # Автоматически применяем все адаптеры (LoRA и т.д.)
        for adapter in self.children.get("adapters", []):
            if hasattr(adapter, "apply"):
                output = adapter.apply(output, self)
        
        return output
    
    def inject_adapter(self, adapter: AbstractBlock):
        """Удобный метод для runtime-инъекции."""
        self.attach_slot("adapters", adapter)