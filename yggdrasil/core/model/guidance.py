# yggdrasil/core/model/guidance.py
from __future__ import annotations

import torch
from typing import Optional, Dict, Any
from omegaconf import DictConfig

from yggdrasil.core.block.base import AbstractBlock
from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import Port, InputPort, OutputPort, TensorSpec


@register_block("guidance/abstract")
class AbstractGuidance(AbstractBlock):
    """Абстрактный guidance-блок (CFG, PAG, SAG, FreeU, custom).
    
    Контракт: реализовать ``process()`` (port-based) или ``__call__()`` (legacy).
    
    Порты:
        IN:  model_output, uncond_output (opt), x (opt), t (opt), condition (opt)
        OUT: guided_output
    
    Пример кастомного guidance::
    
        @register_block("guidance/my_custom")
        class MyGuidance(AbstractGuidance):
            block_type = "guidance/my_custom"
            def __init__(self, config):
                super().__init__(config)
                self.scale = float(config.get("scale", 7.5))
            def process(self, **kw):
                cond = kw["model_output"]
                uncond = kw.get("uncond_output", torch.zeros_like(cond))
                guided = uncond + self.scale * (cond - uncond)
                return {"guided_output": guided}
    """
    
    block_type = "guidance/abstract"
    
    @classmethod
    def declare_io(cls) -> dict:
        return {
            "model_output": InputPort("model_output", spec=TensorSpec(space="latent"), description="Raw model output (conditional)"),
            "uncond_output": InputPort("uncond_output", optional=True, description="Unconditional model output (for CFG)"),
            "condition": InputPort("condition", data_type="any", optional=True, description="Condition embedding"),
            "x": InputPort("x", optional=True, description="Current latents"),
            "t": InputPort("t", data_type="tensor", optional=True, description="Current timestep"),
            "guided_output": OutputPort("guided_output", spec=TensorSpec(space="latent"), description="Guided output"),
        }
    
    def process(self, **port_inputs) -> dict:
        """Port-based execution. Override in subclasses for custom guidance."""
        model_output = port_inputs.get("model_output")
        # Try legacy __call__ for backward compat
        try:
            result = self(model_output, **{k: v for k, v in port_inputs.items() if k != "model_output"})
        except NotImplementedError:
            result = model_output  # passthrough
        return {"guided_output": result, "output": result}
    
    def _define_slots(self):
        return {}
    
    def _forward_impl(self, *args, **kwargs) -> torch.Tensor:
        return self(*args, **kwargs)

    def __call__(
        self,
        model_output: torch.Tensor,
        condition: Optional[Dict[str, Any]] = None,
        model: Optional[Any] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """Legacy interface. New code should override process() instead."""
        raise NotImplementedError(
            f"{type(self).__name__} должен реализовать process() или __call__()"
        )