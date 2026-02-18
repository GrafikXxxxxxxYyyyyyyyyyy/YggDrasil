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
                    self.pretrained, torch_dtype=dtype, low_cpu_mem_usage=False
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
            "control_image": InputPort("control_image", description="Conditioning image (depth, canny, etc.)", optional=True),
            "sample": InputPort("sample", description="Noisy latent sample", optional=True),
            "timestep": InputPort("timestep", description="Timestep", optional=True),
            "encoder_hidden_states": InputPort("encoder_hidden_states", 
                                                description="Text embeddings", optional=True),
            "down_block_residuals": OutputPort("down_block_residuals",
                                                description="ControlNet residuals for down blocks"),
            "mid_block_residual": OutputPort("mid_block_residual",
                                              description="ControlNet residual for mid block"),
            "output": OutputPort("output", description="Dict for backbone adapter_features (down_block_additional_residuals, mid_block_additional_residual)"),
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
                "output": {"down_block_additional_residuals": None, "mid_block_additional_residual": None},
            }

        # If still a string (URL/path), load image (e.g. when pipeline resolution failed or input came from another node)
        if isinstance(control_image, str):
            try:
                from yggdrasil.pipeline import load_image_from_url_or_path, _pil_to_tensor
                pil_img = load_image_from_url_or_path(control_image)
                if pil_img is None:
                    raise ValueError(f"Failed to load control image from {control_image!r}")
                control_image = _pil_to_tensor(pil_img)
            except Exception as e:
                logger.warning("ControlNet: could not load control_image from string %r: %s. Using zero residuals.", control_image, e)
                return {
                    "down_block_residuals": None,
                    "mid_block_residual": None,
                    "output": {"down_block_additional_residuals": None, "mid_block_additional_residual": None},
                }

        # Match dtypes: pipeline often sends float32 (e.g. from URL), model is fp16
        target_dtype = next(self.controlnet.parameters()).dtype
        target_device = next(self.controlnet.parameters()).device
        control_image = control_image.to(device=target_device, dtype=target_dtype)
        sample = kw.get("sample")
        timestep = kw.get("timestep")
        # Resize control image to match generation resolution (sample is latents: H/8 x W/8)
        if sample is not None and hasattr(sample, "shape") and len(sample.shape) == 4:
            target_h, target_w = sample.shape[2] * 8, sample.shape[3] * 8
            if control_image.shape[2] != target_h or control_image.shape[3] != target_w:
                control_image = torch.nn.functional.interpolate(
                    control_image, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
        raw_cond = kw.get("encoder_hidden_states")
        # Graph may pass full condition dict (e.g. SDXL: {encoder_hidden_states, added_cond_kwargs}); extract tensor
        if isinstance(raw_cond, dict):
            encoder_hidden_states = raw_cond.get("encoder_hidden_states")
        else:
            encoder_hidden_states = raw_cond
        if encoder_hidden_states is None:
            logger.warning("ControlNet: encoder_hidden_states missing (condition not connected?). Using zero residuals.")
            return {
                "down_block_residuals": None,
                "mid_block_residual": None,
                "output": {"down_block_additional_residuals": None, "mid_block_additional_residual": None},
            }
        if hasattr(encoder_hidden_states, "to"):
            encoder_hidden_states = encoder_hidden_states.to(device=target_device, dtype=target_dtype)
        if sample is not None and hasattr(sample, "to"):
            sample = sample.to(device=target_device, dtype=target_dtype)
        if timestep is not None and hasattr(timestep, "to"):
            timestep = timestep.to(device=target_device)
        added_cond_kwargs = None
        if isinstance(raw_cond, dict):
            added_cond_kwargs = raw_cond.get("added_cond_kwargs")
        if added_cond_kwargs is not None and isinstance(added_cond_kwargs, dict):
            added_cond_kwargs = {k: v.to(device=target_device, dtype=target_dtype) if hasattr(v, "to") else v for k, v in added_cond_kwargs.items()}

        controlnet_kw: Dict[str, Any] = dict(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=control_image,
            conditioning_scale=self.conditioning_scale,
            return_dict=False,
        )
        if added_cond_kwargs is not None:
            controlnet_kw["added_cond_kwargs"] = added_cond_kwargs
        down_residuals, mid_residual = self.controlnet(**controlnet_kw)
        
        out = {
            "down_block_residuals": down_residuals,
            "mid_block_residual": mid_residual,
            "output": {
                "down_block_additional_residuals": down_residuals,
                "mid_block_additional_residual": mid_residual,
            },
        }
        return out
    
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
