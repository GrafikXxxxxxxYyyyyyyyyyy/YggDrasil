# yggdrasil/blocks/guidances/cfg.py
"""Classifier-Free Guidance — чистый port-based блок.

CFG получает оба выхода (cond + uncond) через порты и комбинирует их.
Никаких скрытых зависимостей (_backbone_ref) — поведение определяется
структурой графа.

Формула: guided = uncond + scale * (cond - uncond)
"""
from __future__ import annotations

import torch
from omegaconf import DictConfig
from typing import Optional, Dict, Any

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import InputPort, OutputPort, TensorSpec
from yggdrasil.core.model.guidance import AbstractGuidance


@register_block("guidance/cfg")
class ClassifierFreeGuidance(AbstractGuidance):
    """Classifier-Free Guidance.
    
    Порты:
        IN:  model_output (cond prediction), uncond_output (uncond prediction)
        OUT: guided_output
    
    Граф должен подавать оба входа. Типичная структура::
    
        conditioner --> backbone_cond  --|
                                        |--> CFG --> solver
        null_cond   --> backbone_uncond -|
    
    Если uncond_output не подключён — passthrough (без guidance).
    """
    
    block_type = "guidance/cfg"
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "guidance/cfg"}
        super().__init__(config)
        self.scale = float(self.config.get("scale", 7.5))
    
    @classmethod
    def declare_io(cls) -> dict:
        return {
            "model_output": InputPort("model_output", spec=TensorSpec(space="latent"),
                                      description="Conditional model prediction"),
            "uncond_output": InputPort("uncond_output", spec=TensorSpec(space="latent"),
                                       optional=True,
                                       description="Unconditional model prediction"),
            "guided_output": OutputPort("guided_output", spec=TensorSpec(space="latent"),
                                         description="Guided output"),
        }
    
    def process(self, **port_inputs) -> dict:
        model_output = port_inputs.get("model_output")
        uncond_output = port_inputs.get("uncond_output")
        
        if model_output is None or self.scale <= 1.0 or uncond_output is None:
            # No guidance: passthrough
            return {"guided_output": model_output, "output": model_output}
        
        guided = uncond_output + self.scale * (model_output - uncond_output)
        return {"guided_output": guided, "output": guided}
    
    def _forward_impl(self, *args, **kwargs) -> torch.Tensor:
        return args[0] if args else torch.zeros(1)
    
    def __call__(self, model_output, **kwargs) -> torch.Tensor:
        """Legacy compat."""
        uncond = kwargs.get("uncond_output")
        if uncond is not None and self.scale > 1.0:
            return uncond + self.scale * (model_output - uncond)
        return model_output
