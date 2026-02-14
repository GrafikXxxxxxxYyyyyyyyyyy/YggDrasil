# yggdrasil/core/model/guidance.py
from __future__ import annotations

import torch
from abc import abstractmethod
from typing import Optional, Dict, Any
from omegaconf import DictConfig

from yggdrasil.core.block.base import AbstractBlock
from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import Port, InputPort, OutputPort, TensorSpec


@register_block("guidance/abstract")
class AbstractGuidance(AbstractBlock):
    """Методы guidance (CFG, PAG, FreeU, custom)."""
    
    block_type = "guidance/abstract"
    
    @classmethod
    def declare_io(cls) -> dict:
        return {
            "model_output": InputPort("model_output", spec=TensorSpec(space="latent"), description="Raw model output (conditional)"),
            "condition": InputPort("condition", data_type="any", optional=True, description="Condition embedding (for internal uncond pass)"),
            "uncond_output": InputPort("uncond_output", optional=True, description="Explicit unconditional output (for dual-pass graph)"),
            "x": InputPort("x", optional=True, description="Current latents (for internal uncond pass)"),
            "t": InputPort("t", data_type="tensor", optional=True, description="Current timestep (for internal uncond pass)"),
            "guided_output": OutputPort("guided_output", spec=TensorSpec(space="latent"), description="Guided output"),
        }
    
    def process(self, **port_inputs) -> dict:
        """Graph-mode guidance. Subclasses should override for custom logic."""
        model_output = port_inputs.get("model_output")
        result = self(model_output, **{k: v for k, v in port_inputs.items() if k != "model_output"})
        return {"guided_output": result, "output": result}
    
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