# yggdrasil/blocks/adapters/controlnet.py
"""ControlNet adapter — adds spatial conditioning to diffusion backbones.

Gracefully degrades without diffusers: runs in stub mode if the library
or pretrained model is not available.

Usage::

    # In a graph template
    graph.add_node("controlnet", BlockBuilder.build({
        "type": "adapter/controlnet",
        "control_type": "depth",
        "pretrained": "lllyasviel/control_v11p_sd15_depth",
    }))
    
    # Inject into backbone
    controlnet.inject_into(backbone_block)
"""
from __future__ import annotations

import logging
import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Any, Dict, Optional

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import InputPort, OutputPort
from .base import AbstractAdapter

logger = logging.getLogger(__name__)

# Graceful degradation: try to import diffusers
try:
    from diffusers.models.controlnets import ControlNetModel
    _HAS_DIFFUSERS_CONTROLNET = True
except ImportError:
    _HAS_DIFFUSERS_CONTROLNET = False
    logger.debug("diffusers ControlNetModel not available — stub mode")


@register_block("adapter/controlnet")
class ControlNetAdapter(AbstractAdapter):
    """ControlNet adapter with graceful degradation.
    
    If diffusers is available and pretrained model is specified, loads real weights.
    Otherwise, operates in stub mode (passthrough).
    """
    
    block_type = "adapter/controlnet"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.control_type = config.get("control_type", "depth")
        self.pretrained = config.get(
            "pretrained",
            f"lllyasviel/control_v11p_sd15_{self.control_type}"
        )
        self.conditioning_scale = config.get("conditioning_scale", 1.0)
        self.controlnet: Optional[nn.Module] = None
        
        # Try to load real model
        if _HAS_DIFFUSERS_CONTROLNET and config.get("pretrained"):
            try:
                dtype = torch.float16 if config.get("fp16", True) else torch.float32
                self.controlnet = ControlNetModel.from_pretrained(
                    self.pretrained, torch_dtype=dtype,
                )
                self.controlnet.requires_grad_(config.get("trainable", False))
                logger.info(f"Loaded ControlNet: {self.pretrained}")
            except Exception as e:
                logger.warning(f"Failed to load ControlNet '{self.pretrained}': {e}. Using stub mode.")
                self.controlnet = None
        
        if self.controlnet is None:
            logger.info(f"ControlNet '{self.control_type}' in stub mode (no weights loaded)")
    
    @classmethod
    def declare_io(cls):
        return {
            "control_image": InputPort("control_image", description="Conditioning image (depth, canny, etc.)"),
            "sample": InputPort("sample", description="Noisy latent sample", optional=True),
            "timestep": InputPort("timestep", description="Timestep", optional=True),
            "encoder_hidden_states": InputPort("encoder_hidden_states", 
                                                description="Text embeddings", optional=True),
            "down_block_residuals": OutputPort("down_block_residuals",
                                                description="ControlNet residuals for down blocks"),
            "mid_block_residual": OutputPort("mid_block_residual",
                                              description="ControlNet residual for mid block"),
        }
    
    def process(self, **kw) -> Dict[str, Any]:
        """Process control image through ControlNet.
        
        Returns residuals that should be added to the backbone's intermediate features.
        """
        control_image = kw.get("control_image")
        if control_image is None or self.controlnet is None:
            # Stub: return zero residuals
            return {
                "down_block_residuals": None,
                "mid_block_residual": None,
            }
        
        sample = kw.get("sample")
        timestep = kw.get("timestep")
        encoder_hidden_states = kw.get("encoder_hidden_states")
        
        down_residuals, mid_residual = self.controlnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=control_image,
            conditioning_scale=self.conditioning_scale,
            return_dict=False,
        )
        
        return {
            "down_block_residuals": down_residuals,
            "mid_block_residual": mid_residual,
        }
    
    def inject_into(self, target):
        """Inject ControlNet conditioning into a backbone.
        
        This modifies the backbone's forward pass to include ControlNet residuals.
        """
        if self.controlnet is None:
            logger.warning("ControlNet inject_into: no model loaded, injection is a no-op")
            return
        
        if hasattr(target, "unet"):
            original_forward = target.unet.forward
            controlnet = self.controlnet
            scale = self.conditioning_scale
            
            def wrapped_forward(*args, **kwargs):
                condition = kwargs.get("condition", {})
                control_image = condition.get("control_image")
                if control_image is not None:
                    down_residuals, mid_residual = controlnet(
                        sample=args[0] if args else kwargs.get("sample"),
                        timestep=kwargs.get("timestep"),
                        encoder_hidden_states=condition.get("encoder_hidden_states"),
                        controlnet_cond=control_image,
                        conditioning_scale=scale,
                        return_dict=False,
                    )
                    kwargs["down_block_additional_residuals"] = down_residuals
                    kwargs["mid_block_additional_residual"] = mid_residual
                return original_forward(*args, **kwargs)
            
            target.unet.forward = wrapped_forward
        else:
            target.add_pre_hook(self._controlnet_hook)
    
    def _controlnet_hook(self, module, *args, **kwargs):
        """Hook for custom backbones."""
        condition = kwargs.get("condition", {})
        control_image = condition.get("control_image")
        if control_image is not None and self.controlnet is not None:
            sample = args[0] if args else kwargs.get("x")
            timestep = kwargs.get("timestep")
            encoder_hidden_states = condition.get("encoder_hidden_states")
            down_residuals, mid_residual = self.controlnet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=control_image,
                conditioning_scale=self.conditioning_scale,
                return_dict=False,
            )
            condition["down_block_residuals"] = down_residuals
            condition["mid_block_residual"] = mid_residual
    
    def apply(self, output: torch.Tensor, context=None) -> torch.Tensor:
        return output
