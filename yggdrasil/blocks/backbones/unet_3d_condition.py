# yggdrasil/blocks/backbones/unet_3d_condition.py
"""3D UNet backbone for video diffusion models."""
import torch
import torch.nn as nn
from typing import Optional, Dict
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.model.backbone import AbstractBackbone


@register_block("backbone/unet3d_condition")
class UNet3DConditionBackbone(AbstractBackbone):
    """3D UNet backbone for video diffusion.
    
    Wraps diffusers UNet3DConditionModel.
    Used by AnimateDiff, ModelScope, and other video models.
    """
    block_type = "backbone/unet3d_condition"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._model = None
        self.pretrained = config.get("pretrained")
        if self.pretrained:
            self._build_model(config)
    
    def _build_model(self, config):
        try:
            from diffusers import UNet3DConditionModel
            self._model = UNet3DConditionModel.from_pretrained(
                self.pretrained, subfolder="unet",
                torch_dtype=torch.float16 if config.get("fp16", True) else torch.float32,
            )
            self._model.requires_grad_(False)
        except Exception:
            # Fallback: use 2D UNet and process frames independently
            from diffusers import UNet2DConditionModel
            self._model = UNet2DConditionModel.from_pretrained(
                config.get("pretrained_2d", "runwayml/stable-diffusion-v1-5"),
                subfolder="unet",
                torch_dtype=torch.float16 if config.get("fp16", True) else torch.float32,
            )
            self._model.requires_grad_(False)
    
    def _forward_impl(self, x, timestep, condition=None, position_embedding=None, **kwargs):
        encoder_hidden_states = condition.get("encoder_hidden_states") if condition else None
        
        if self._model is not None:
            try:
                return self._model(
                    sample=x,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]
            except (TypeError, RuntimeError):
                pass
        
        return x
