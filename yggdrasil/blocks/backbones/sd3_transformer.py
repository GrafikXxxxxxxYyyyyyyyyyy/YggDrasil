# yggdrasil/blocks/backbones/sd3_transformer.py
"""SD3 Transformer backbone.

Wraps diffusers SD3Transformer2DModel for direct use in ComputeGraph.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.model.backbone import AbstractBackbone


@register_block("backbone/sd3_transformer")
class SD3TransformerBackbone(AbstractBackbone):
    """SD3 Transformer backbone.
    
    Wraps diffusers SD3Transformer2DModel with flow matching.
    Falls back to MMDiT backbone if diffusers model unavailable.
    """
    block_type = "backbone/sd3_transformer"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._model = None
        self.pretrained = config.get("pretrained", "stabilityai/stable-diffusion-3-medium")
        self._build_model(config)
    
    def _build_model(self, config):
        try:
            from diffusers import SD3Transformer2DModel
            self._model = SD3Transformer2DModel.from_pretrained(
                self.pretrained, subfolder="transformer",
                torch_dtype=torch.float16 if config.get("fp16", True) else torch.float32,
            )
            self._model.requires_grad_(False)
        except Exception:
            from .mmdit import MMDiTBackbone
            fallback_config = dict(config)
            fallback_config.setdefault("hidden_dim", 1536)
            fallback_config.setdefault("num_layers", 24)
            fallback_config.setdefault("num_heads", 24)
            fallback_config.setdefault("in_channels", 16)
            fallback_config.setdefault("patch_size", 2)
            self._model = MMDiTBackbone(DictConfig(fallback_config))
    
    def _forward_impl(self, x, timestep, condition=None, position_embedding=None, **kwargs):
        encoder_hidden_states = condition.get("encoder_hidden_states") if condition else None
        pooled_projections = condition.get("pooled_embedding") if condition else None
        
        if hasattr(self._model, 'forward') and not isinstance(self._model, AbstractBackbone):
            try:
                return self._model(
                    hidden_states=x,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled_projections,
                    return_dict=False,
                )[0]
            except (TypeError, RuntimeError):
                pass
        
        if hasattr(self._model, '_forward_impl'):
            return self._model._forward_impl(x, timestep, condition, position_embedding, **kwargs)
        
        return x
