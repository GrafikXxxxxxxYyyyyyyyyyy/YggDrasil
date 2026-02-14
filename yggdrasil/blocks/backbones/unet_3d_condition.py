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
            model_dtype = next(self._model.parameters()).dtype
            model_device = next(self._model.parameters()).device
            if x.dtype != model_dtype:
                x = x.to(dtype=model_dtype, device=model_device)
            if encoder_hidden_states is not None and encoder_hidden_states.dtype != model_dtype:
                encoder_hidden_states = encoder_hidden_states.to(dtype=model_dtype, device=model_device)
            if timestep.dtype != model_dtype and timestep.dtype != torch.int64:
                timestep = timestep.to(dtype=model_dtype, device=model_device)
        
        # Adapter features (ControlNet/T2I) — pass through when model supports it
        af = kwargs.get("adapter_features")
        down_block_residuals = None
        mid_block_residual = None
        if isinstance(af, dict):
            down_block_residuals = af.get("down_block_additional_residuals") or af.get("down_block_residuals")
            mid_block_residual = af.get("mid_block_additional_residual") or af.get("mid_block_residual")
        elif isinstance(af, (tuple, list)) and len(af) >= 2 and not (af and isinstance(af[0], dict)):
            down_block_residuals, mid_block_residual = af[0], af[1]
        elif isinstance(af, list) and af and isinstance(af[0], dict):
            all_down = [a.get("down_block_additional_residuals") or a.get("down_block_residuals") for a in af]
            all_down = [x for x in all_down if x is not None]
            all_mid = [a.get("mid_block_additional_residual") or a.get("mid_block_residual") for a in af]
            all_mid = [x for x in all_mid if x is not None]
            if all_down:
                down_block_residuals = [sum(t) for t in zip(*all_down)]
            if all_mid:
                mid_block_residual = sum(all_mid)

        if self._model is not None:
            try:
                unet_kw = dict(
                    sample=x,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )
                if down_block_residuals is not None:
                    unet_kw["down_block_additional_residuals"] = down_block_residuals
                if mid_block_residual is not None:
                    unet_kw["mid_block_additional_residual"] = mid_block_residual
                out = self._model(**unet_kw)[0]
                return out
            except (TypeError, RuntimeError) as e:
                # 5D input (B, C, T, H, W) with 2D UNet fallback: process frames as batch
                if x.dim() == 5 and hasattr(self._model, "forward"):
                    B, C, T, H, W = x.shape
                    x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                    if encoder_hidden_states is not None and encoder_hidden_states.dim() >= 2:
                        enc = encoder_hidden_states.repeat_interleave(T, dim=0)
                    else:
                        enc = encoder_hidden_states
                    if timestep.dim() >= 1:
                        t = timestep.repeat_interleave(T, dim=0)
                    else:
                        t = timestep
                    out_2d = self._model(sample=x_2d, timestep=t, encoder_hidden_states=enc, return_dict=False)[0]
                    out = out_2d.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
                    return out
                raise
        return x
