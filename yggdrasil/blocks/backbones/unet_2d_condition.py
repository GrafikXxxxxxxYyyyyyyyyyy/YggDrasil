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
        
        return self.unet(
            sample=x,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )[0]