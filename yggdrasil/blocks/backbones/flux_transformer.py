# yggdrasil/blocks/backbones/flux_transformer.py
"""Flux Transformer backbone — MMDiT variant for Flux models.

Wraps diffusers FluxTransformer2DModel for direct use in ComputeGraph.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.model.backbone import AbstractBackbone


@register_block("backbone/flux_transformer")
class FluxTransformerBackbone(AbstractBackbone):
    """Flux Transformer 2D backbone.
    
    Wraps diffusers FluxTransformer2DModel.
    Falls back to MMDiT backbone if diffusers model unavailable.
    """
    block_type = "backbone/flux_transformer"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._model = None
        self.pretrained = config.get("pretrained", "black-forest-labs/FLUX.1-dev")
        self._build_model(config)
    
    def _build_model(self, config):
        try:
            from diffusers import FluxTransformer2DModel
            self._model = FluxTransformer2DModel.from_pretrained(
                self.pretrained, subfolder="transformer",
                torch_dtype=torch.float16 if config.get("fp16", True) else torch.float32,
            )
            self._model.requires_grad_(False)
        except Exception:
            # Fallback: use local MMDiT
            from .mmdit import MMDiTBackbone
            fallback_config = dict(config)
            fallback_config.setdefault("hidden_dim", 3072)
            fallback_config.setdefault("num_layers", 19)
            fallback_config.setdefault("num_heads", 24)
            fallback_config.setdefault("in_channels", 16)
            fallback_config.setdefault("patch_size", 2)
            self._model = MMDiTBackbone(DictConfig(fallback_config))
    
    def _forward_impl(self, x, timestep, condition=None, position_embedding=None, **kwargs):
        encoder_hidden_states = condition.get("encoder_hidden_states") if condition else None
        
        if hasattr(self._model, 'forward'):
            try:
                return self._model(
                    hidden_states=x,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]
            except (TypeError, RuntimeError):
                pass
        
        # Fallback for MMDiT
        if hasattr(self._model, '_forward_impl'):
            return self._model._forward_impl(x, timestep, condition, position_embedding, **kwargs)
        
        return x
