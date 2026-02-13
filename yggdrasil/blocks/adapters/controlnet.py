# yggdrasil/blocks/adapters/controlnet.py
from __future__ import annotations

import torch
from omegaconf import DictConfig
from typing import Dict, Any

from diffusers.models.controlnets import ControlNetModel

from yggdrasil.core.block.registry import register_block
from .base import AbstractAdapter
from yggdrasil.core.model.backbone import AbstractBackbone


@register_block("adapter/controlnet")
class ControlNetAdapter(AbstractAdapter):
    """ControlNet адаптер."""
    
    block_type = "adapter/controlnet"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.control_type = config.get("control_type", "depth")
        self.pretrained = config.get(
            "pretrained",
            f"lllyasviel/control_v11p_sd15_{self.control_type}"
        )
        
        self.controlnet = ControlNetModel.from_pretrained(
            self.pretrained,
            torch_dtype=torch.float16 if config.get("fp16", True) else torch.float32,
        )
        self.controlnet.requires_grad_(config.get("trainable", False))
    
    def inject_into(self, target: AbstractBackbone):
        if hasattr(target, "unet"):
            original_forward = target.unet.forward
            
            def wrapped_forward(*args, **kwargs):
                condition = kwargs.get("condition", {})
                control_image = condition.get("control_image")
                if control_image is not None:
                    control = self.controlnet(
                        sample=args[0] if args else kwargs.get("sample"),
                        timestep=kwargs.get("timestep"),
                        encoder_hidden_states=condition.get("encoder_hidden_states"),
                        controlnet_cond=control_image,
                        return_dict=False
                    )[0]
                    condition["control"] = control
                return original_forward(*args, **kwargs)
            
            target.unet.forward = wrapped_forward
        else:
            target.add_pre_hook(self._controlnet_hook)
    
    def _controlnet_hook(self, module, x, timestep, condition: Dict[str, Any], **kwargs):
        """Хук для кастомных backbones."""
        control_image = condition.get("control_image")
        if control_image is not None:
            control = self.controlnet(
                x,
                timestep,
                encoder_hidden_states=condition.get("encoder_hidden_states"),
                controlnet_cond=control_image,
                return_dict=False
            )[0]
            condition["control"] = control
        return x, timestep, condition 
    
    def apply(self, output: torch.Tensor, context=None) -> torch.Tensor:
        return output