from __future__ import annotations

import torch 

from abc import abstractmethod
from typing import Dict, Any, Optional
from omegaconf import DictConfig

from ...core.block.base import AbstractBaseBlock
from ...core.block.registry import register_block
from ...core.block.port import Port, InputPort, OutputPort, TensorSpec
from ...core.model.backbone import AbstractBackbone


@register_block("adapter/abstract")
class AbstractAdapter(AbstractBaseBlock):
    """Базовый адаптер — Lego-кирпичик для модификации модели."""
    
    block_type = "adapter/abstract"
    
    @classmethod
    def declare_io(cls) -> dict:
        return {
            "input": InputPort("input", data_type="any", description="Input to adapt"),
            "context": InputPort("context", data_type="any", optional=True, description="Context"),
            "output": OutputPort("output", data_type="any", description="Adapted output"),
        }
    
    def process(self, **port_inputs) -> dict:
        x = port_inputs.get("input", port_inputs.get("x"))
        context = port_inputs.get("context")
        result = self.apply(x, context)
        return {"output": result}
    
    @abstractmethod
    def inject_into(self, target: AbstractBackbone | ModularDiffusionModel):
        """Инжектирует себя в целевую модель/backbone."""
        pass
    
    @abstractmethod
    def apply(self, output: torch.Tensor, context: Any = None) -> torch.Tensor:
        """Применяется на каждом forward (если нужно)."""
        return output