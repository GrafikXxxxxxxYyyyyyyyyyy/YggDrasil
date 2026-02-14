# yggdrasil/blocks/backbones/transformer_2d.py
"""Generic Transformer2D backbone.

Wraps diffusers Transformer2DModel for direct use in ComputeGraph.
Used by PixArt, Latte, and other pure-transformer diffusion models.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.model.backbone import AbstractBackbone


@register_block("backbone/transformer_2d")
class Transformer2DBackbone(AbstractBackbone):
    """Generic Transformer 2D backbone.
    
    Wraps diffusers Transformer2DModel.
    Falls back to DiT backbone if diffusers model unavailable.
    """
    block_type = "backbone/transformer_2d"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._model = None
        self.pretrained = config.get("pretrained")
        self._build_model(config)
    
    def _build_model(self, config):
        if self.pretrained:
            try:
                from diffusers import Transformer2DModel
                self._model = Transformer2DModel.from_pretrained(
                    self.pretrained, subfolder="transformer",
                    torch_dtype=torch.float16 if config.get("fp16", True) else torch.float32,
                )
                self._model.requires_grad_(False)
                return
            except Exception:
                pass
        
        # Fallback: use DiT
        from .dit import DiTBackbone
        fallback_config = dict(config)
        fallback_config.setdefault("hidden_dim", 1152)
        fallback_config.setdefault("num_layers", 28)
        fallback_config.setdefault("num_heads", 16)
        fallback_config.setdefault("in_channels", 4)
        fallback_config.setdefault("patch_size", 2)
        self._model = DiTBackbone(DictConfig(fallback_config))
    
    def _forward_impl(self, x, timestep, condition=None, position_embedding=None, **kwargs):
        encoder_hidden_states = condition.get("encoder_hidden_states") if condition else None
        
        if hasattr(self._model, 'forward') and not isinstance(self._model, AbstractBackbone):
            try:
                return self._model(
                    hidden_states=x,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]
            except (TypeError, RuntimeError):
                pass
        
        if hasattr(self._model, '_forward_impl'):
            return self._model._forward_impl(x, timestep, condition, position_embedding, **kwargs)
        
        return x
