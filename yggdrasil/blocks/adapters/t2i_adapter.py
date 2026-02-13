# yggdrasil/blocks/adapters/t2i_adapter.py
from __future__ import annotations

import torch
from omegaconf import DictConfig
from typing import Optional, Dict, Any

from diffusers import T2IAdapter
from diffusers.models import UNet2DConditionModel   # ← ИСПРАВЛЕНО здесь

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.slot import Slot
from .base import AbstractAdapter
from yggdrasil.core.model.backbone import AbstractBackbone


@register_block("adapter/t2i")
class T2IAdapter(AbstractAdapter):
    """T2I-Adapter — лёгкий структурный контроллер."""
    
    block_type = "adapter/t2i"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.adapter_type = config.get("adapter_type", "depth")
        self.pretrained = config.get(
            "pretrained",
            f"TencentARC/t2i-adapter-{self.adapter_type}-sd15v2"
        )
        self.scale = config.get("scale", 1.0)
        
        self.t2i_adapter = T2IAdapter.from_pretrained(
            self.pretrained,
            torch_dtype=torch.float16 if config.get("fp16", True) else torch.float32,
        )
        self.t2i_adapter.requires_grad_(config.get("trainable", False))
        
        self.adapter_features: Optional[torch.Tensor] = None
    
    def _define_slots(self):
        return {
            "conditioner": Slot(
                name="conditioner",
                accepts=AbstractBlock,
                multiple=True,
                optional=True
            )
        }
    
    def inject_into(self, target: AbstractBackbone):
        if hasattr(target, "unet") and isinstance(target.unet, UNet2DConditionModel):
            original_forward = target.unet.forward
            
            def t2i_forward(*args, **kwargs):
                if self.adapter_features is not None:
                    kwargs["adapter_features"] = self.adapter_features
                return original_forward(*args, **kwargs)
            
            target.unet.forward = t2i_forward
        else:
            target.add_pre_hook(self._pre_forward_hook)
    
    def _pre_forward_hook(self, module, x, timestep, condition: Dict[str, Any], **kwargs):
        """Хук перед forward backbone."""
        control_image = condition.get("control_image")
        if control_image is None:
            return x, timestep, condition
        
        if control_image.max() > 1.0:
            control_image = (control_image + 1.0) / 2.0
        
        with torch.no_grad():
            self.adapter_features = self.t2i_adapter(control_image)
        
        condition["adapter_features"] = self.adapter_features * self.scale
        return x, timestep, condition
    
    def apply(self, output: torch.Tensor, context=None) -> torch.Tensor:
        return output
    
    def __repr__(self):
        return f"<T2IAdapter type={self.adapter_type} scale={self.scale}>"