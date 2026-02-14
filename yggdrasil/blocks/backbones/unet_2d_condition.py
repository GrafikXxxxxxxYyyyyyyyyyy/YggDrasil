import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.model.backbone import AbstractBackbone


@register_block("backbone/unet2d_condition")
class UNet2DConditionBackbone(AbstractBackbone):
    """Нативная обёртка UNet2DConditionModel из diffusers (SD 1.5 / SDXL)."""
    
    block_type = "backbone/unet2d_condition"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.unet = UNet2DConditionModel.from_pretrained(
            config.get("pretrained", "runwayml/stable-diffusion-v1-5"),
            subfolder="unet",
            torch_dtype=torch.float16 if config.get("fp16", True) else torch.float32,
        )
        self.unet.requires_grad_(False)  # по умолчанию заморожен
    
    def _forward_impl(
        self,
        x: torch.Tensor,                    # [B, 4, 64, 64]
        timestep: torch.Tensor,             # [B]
        condition: dict | None = None,      # {"encoder_hidden_states": [B, 77, 768]}
        **kwargs
    ) -> torch.Tensor:
        encoder_hidden_states = condition.get("encoder_hidden_states") if condition else None
        
        # Determine model device and dtype
        model_device = next(self.unet.parameters()).device
        model_dtype = next(self.unet.parameters()).dtype
        
        # Ensure all inputs are on the same device as the model
        if x.device != model_device:
            x = x.to(model_device)
        if timestep.device != model_device:
            timestep = timestep.to(model_device)
        if encoder_hidden_states is not None and encoder_hidden_states.device != model_device:
            encoder_hidden_states = encoder_hidden_states.to(model_device)
        
        # Cast to model dtype
        if encoder_hidden_states is not None and encoder_hidden_states.dtype != model_dtype:
            encoder_hidden_states = encoder_hidden_states.to(dtype=model_dtype)
        if x.dtype != model_dtype:
            x = x.to(dtype=model_dtype)
        
        # Pass through ControlNet/Adapter residuals (from condition dict, kwargs, or adapter_features port)
        down_block_residuals = kwargs.get("down_block_additional_residuals")
        mid_block_residual = kwargs.get("mid_block_additional_residual")
        if down_block_residuals is None and condition is not None and isinstance(condition, dict):
            down_block_residuals = condition.get("down_block_additional_residuals")
            mid_block_residual = condition.get("mid_block_additional_residual")
        if down_block_residuals is None:
            af = kwargs.get("adapter_features")
            if isinstance(af, dict):
                down_block_residuals = af.get("down_block_additional_residuals") or af.get("down_block_residuals")
                mid_block_residual = af.get("mid_block_additional_residual") or af.get("mid_block_residual")
            elif isinstance(af, (tuple, list)) and len(af) >= 2:
                down_block_residuals, mid_block_residual = af[0], af[1]

        added_cond_kwargs = None
        if condition is not None and isinstance(condition, dict):
            added_cond_kwargs = condition.get("added_cond_kwargs")
        if added_cond_kwargs is None:
            added_cond_kwargs = kwargs.get("added_cond_kwargs")

        unet_kw = dict(
            sample=x,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_residuals,
            mid_block_additional_residual=mid_block_residual,
            return_dict=False,
        )
        if added_cond_kwargs is not None:
            unet_kw["added_cond_kwargs"] = added_cond_kwargs
        return self.unet(**unet_kw)[0]