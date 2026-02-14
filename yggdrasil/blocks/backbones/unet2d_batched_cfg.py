# yggdrasil/blocks/backbones/unet2d_batched_cfg.py
"""Batched UNet + CFG in one forward (diffusers parity).

Single UNet call with latent_model_input = cat([latents, latents]) and
encoder_hidden_states = cat([negative_embeds, positive_embeds]), then chunk + CFG + optional rescale.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from omegaconf import DictConfig
from typing import Dict, Any

from ...core.block.registry import register_block
from ...core.block.port import InputPort, OutputPort, TensorSpec
from ...core.model.backbone import AbstractBackbone
from ..guidances.cfg import _rescale_noise_cfg


@register_block("backbone/unet2d_batched_cfg")
class UNet2DBatchedCFGBackbone(AbstractBackbone):
    """Single batched UNet forward with CFG + optional guidance_rescale (diffusers parity)."""

    block_type = "backbone/unet2d_batched_cfg"

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.unet = UNet2DConditionModel.from_pretrained(
            config.get("pretrained", "runwayml/stable-diffusion-v1-5"),
            subfolder="unet",
            torch_dtype=torch.float16 if config.get("fp16", True) else torch.float32,
        )
        self.unet.requires_grad_(False)
        self.scale = float(config.get("scale", 7.5))
        self.guidance_rescale = float(config.get("guidance_rescale", 0.0))

    @classmethod
    def declare_io(cls) -> dict:
        return {
            "x": InputPort("x", spec=TensorSpec(space="latent"), description="Latent input"),
            "timestep": InputPort("timestep", data_type="tensor", description="Timestep"),
            "condition": InputPort("condition", data_type="dict", description="Condition embeddings (positive)"),
            "uncond": InputPort("uncond", data_type="dict", description="Unconditional embeddings (negative)"),
            "output": OutputPort("output", spec=TensorSpec(space="latent"), description="Guided noise prediction"),
        }

    def process(self, **port_inputs) -> Dict[str, Any]:
        x = port_inputs["x"]
        timestep = port_inputs["timestep"]
        condition = port_inputs.get("condition")
        uncond = port_inputs.get("uncond")
        if condition is None:
            condition = {}
        if uncond is None:
            uncond = {}

        cond_emb = condition.get("encoder_hidden_states") if isinstance(condition, dict) else condition
        uncond_emb = uncond.get("encoder_hidden_states") if isinstance(uncond, dict) else uncond
        if cond_emb is None or uncond_emb is None:
            raise ValueError("backbone/unet2d_batched_cfg requires condition and uncond with encoder_hidden_states")

        model_device = next(self.unet.parameters()).device
        model_dtype = next(self.unet.parameters()).dtype

        # Batched like diffusers: [uncond, cond]
        B = x.shape[0]
        latent_2 = torch.cat([x, x], dim=0)
        emb_2 = torch.cat([uncond_emb, cond_emb], dim=0)
        # Same timestep for both halves (diffusers passes one t for the whole batch)
        if timestep.numel() == 1:
            timestep_2 = timestep.flatten().expand(2 * B)
        else:
            timestep_2 = timestep.repeat_interleave(2)

        latent_2 = latent_2.to(device=model_device, dtype=model_dtype)
        emb_2 = emb_2.to(device=model_device, dtype=model_dtype)
        timestep_2 = timestep_2.to(device=model_device)

        noise_pred = self.unet(
            sample=latent_2,
            timestep=timestep_2,
            encoder_hidden_states=emb_2,
            return_dict=False,
        )[0]

        uncond_pred, cond_pred = noise_pred.chunk(2, dim=0)
        guided = uncond_pred + self.scale * (cond_pred - uncond_pred)
        if self.guidance_rescale > 0.0:
            guided = _rescale_noise_cfg(guided, cond_pred, self.guidance_rescale)

        return {"output": guided}
