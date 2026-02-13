# yggdrasil/blocks/guidances/cfg.py
from __future__ import annotations

import torch
from omegaconf import DictConfig
from typing import Optional, Dict, Any

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.model.guidance import AbstractGuidance
from yggdrasil.core.model.modular import ModularDiffusionModel


@register_block("guidance/cfg")
class ClassifierFreeGuidance(AbstractGuidance):
    """Classifier-Free Guidance (работает с любой моделью)."""
    
    block_type = "guidance/cfg"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.scale = config.get("scale", 7.5)
        self.null_condition = config.get("null_condition", {"text": ""})
    
    # Обязательный метод из AbstractBlock
    def _forward_impl(self, *args, **kwargs) -> torch.Tensor:
        """Заглушка (реальная логика в __call__)."""
        return args[0] if args else torch.zeros(1)
    
    def __call__(
        self,
        model_output: torch.Tensor,
        condition: Optional[Dict[str, Any]] = None,
        model: Optional[ModularDiffusionModel] = None,
        x: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Применяем CFG."""
        if condition is None or model is None or x is None or t is None or self.scale <= 1.0:
            return model_output
        
        # Отключаем guidance на время uncond прохода
        original_guidances = model.children.get("guidance", [])
        model.children["guidance"] = []
        
        try:
            null_condition = condition.copy()
            for key in ["text"]:
                if key in null_condition:
                    null_condition[key] = self.null_condition.get(key, "")
            
            uncond_output = model._forward_impl(
                x=x,
                t=t,
                condition=null_condition,
                return_dict=False
            )
            
            guided = uncond_output + self.scale * (model_output - uncond_output)
            return guided
            
        finally:
            model.children["guidance"] = original_guidances