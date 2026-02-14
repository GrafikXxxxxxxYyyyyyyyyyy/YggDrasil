# yggdrasil/blocks/guidances/cfg.py
"""Classifier-Free Guidance — работает и в legacy, и в graph mode.

В graph mode поддерживает три стратегии:
1. Explicit dual-pass: получает uncond_output через порт (пользователь сам собрал граф)
2. Internal dual-pass: имеет _backbone_ref и вызывает backbone повторно с null condition
3. Passthrough: если ничего не доступно — пропускает без изменений
"""
from __future__ import annotations

import torch
from omegaconf import DictConfig
from typing import Optional, Dict, Any

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.model.guidance import AbstractGuidance


@register_block("guidance/cfg")
class ClassifierFreeGuidance(AbstractGuidance):
    """Classifier-Free Guidance (работает с любой моделью в любом режиме)."""
    
    block_type = "guidance/cfg"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.scale = config.get("scale", 7.5)
        self.null_condition = config.get("null_condition", {"text": ""})
        # Backbone reference for internal dual-pass in graph mode
        self._backbone_ref = None
    
    def _forward_impl(self, *args, **kwargs) -> torch.Tensor:
        return args[0] if args else torch.zeros(1)
    
    def process(self, **port_inputs) -> dict:
        """Graph-mode CFG с тремя стратегиями."""
        model_output = port_inputs.get("model_output")
        
        if model_output is None or self.scale <= 1.0:
            return {"guided_output": model_output, "output": model_output}
        
        # Strategy 1: explicit uncond_output from graph
        uncond_output = port_inputs.get("uncond_output")
        if uncond_output is not None:
            guided = self._apply_cfg(model_output, uncond_output)
            return {"guided_output": guided, "output": guided}
        
        # Strategy 2: internal dual-pass via backbone reference
        x = port_inputs.get("x")
        t = port_inputs.get("t")
        condition = port_inputs.get("condition")
        
        if self._backbone_ref is not None and x is not None and t is not None:
            null_cond = self._make_null_condition(condition)
            with torch.no_grad():
                uncond_result = self._backbone_ref.process(
                    x=x, timestep=t, condition=null_cond,
                )
                uncond_out = uncond_result.get("output")
            
            if uncond_out is not None:
                guided = self._apply_cfg(model_output, uncond_out)
                return {"guided_output": guided, "output": guided}
        
        # Strategy 3: passthrough (warning — no guidance applied)
        return {"guided_output": model_output, "output": model_output}
    
    def _apply_cfg(self, cond_output: torch.Tensor, uncond_output: torch.Tensor) -> torch.Tensor:
        """Core CFG formula: uncond + scale * (cond - uncond)."""
        return uncond_output + self.scale * (cond_output - uncond_output)
    
    def _make_null_condition(self, condition):
        """Create null/empty condition for unconditional pass."""
        if condition is None:
            return None
        
        if isinstance(condition, torch.Tensor):
            return torch.zeros_like(condition)
        
        if isinstance(condition, dict):
            null = {}
            for key, value in condition.items():
                if isinstance(value, torch.Tensor):
                    null[key] = torch.zeros_like(value)
                elif isinstance(value, str):
                    null[key] = ""
                else:
                    null[key] = value
            return null
        
        return None
    
    def __call__(
        self,
        model_output: torch.Tensor,
        condition: Optional[Dict[str, Any]] = None,
        model: Optional[Any] = None,
        x: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Legacy mode CFG (slot-based pipeline via ModularDiffusionModel)."""
        if condition is None or model is None or x is None or t is None or self.scale <= 1.0:
            return model_output
        
        # Legacy path: disable guidance, run model with null condition
        original_guidances = model._slot_children.get("guidance", [])
        model._slot_children["guidance"] = []
        
        try:
            null_condition = condition.copy() if isinstance(condition, dict) else {}
            for key in ["text"]:
                if key in null_condition:
                    null_condition[key] = self.null_condition.get(key, "")
            
            uncond_output = model._forward_impl(
                x=x, t=t, condition=null_condition, return_dict=False
            )
            
            return self._apply_cfg(model_output, uncond_output)
        finally:
            model._slot_children["guidance"] = original_guidances
