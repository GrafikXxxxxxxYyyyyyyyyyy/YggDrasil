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
            "image_prompt_embeds": InputPort("image_prompt_embeds", data_type="any", optional=True, description="IP-Adapter image embeddings (batch, num_tokens, dim)"),
            "adapter_features": InputPort("adapter_features", data_type="any", optional=True, description="ControlNet/Adapter residuals (dict or tuple)"),
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

        added_cond = condition.get("added_cond_kwargs") if isinstance(condition, dict) else None
        added_uncond = uncond.get("added_cond_kwargs") if isinstance(uncond, dict) else None

        model_device = next(self.unet.parameters()).device
        model_dtype = next(self.unet.parameters()).dtype

        # Batched like diffusers: [uncond, cond]
        B = x.shape[0]
        latent_2 = torch.cat([x, x], dim=0)
        emb_2 = torch.cat([uncond_emb, cond_emb], dim=0)
        # Same timestep for both halves. Pass as (1,) then UNet expands to batch — avoids "Boolean value of Tensor" in some diffusers code paths.
        if timestep.numel() == 1:
            timestep_2 = timestep.flatten().to(device=model_device)
        else:
            t_flat = timestep.flatten().to(device=model_device)
            if t_flat.numel() > 1 and (t_flat[0] == t_flat).all().item():
                timestep_2 = t_flat[0:1]
            else:
                timestep_2 = t_flat.repeat_interleave(2)

        latent_2 = latent_2.to(device=model_device, dtype=model_dtype)
        emb_2 = emb_2.to(device=model_device, dtype=model_dtype)
        if timestep_2.dim() == 0:
            timestep_2 = timestep_2.unsqueeze(0)

        # IP-Adapter: если передан image — (text, image), иначе только text (пайплайн без IP-Adapter)
        image_embeds = port_inputs.get("image_prompt_embeds")
        if image_embeds is not None:
            image_embeds = image_embeds.to(device=model_device, dtype=model_dtype)
            image_embeds_2 = torch.cat([image_embeds, image_embeds], dim=0)  # same for [uncond, cond]
            encoder_hidden_states = (emb_2, image_embeds_2)
        else:
            # Нет ip_image: pipeline без IP-Adapter — один тензор (UNet с обычными процессорами)
            encoder_hidden_states = emb_2

        unet_kw = dict(
            sample=latent_2,
            timestep=timestep_2,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )
        if added_cond is not None and added_uncond is not None:
            text_embeds_2 = torch.cat([added_uncond["text_embeds"], added_cond["text_embeds"]], dim=0)
            time_ids_2 = torch.cat([added_uncond["time_ids"], added_cond["time_ids"]], dim=0)
            unet_kw["added_cond_kwargs"] = {"text_embeds": text_embeds_2.to(model_device), "time_ids": time_ids_2.to(model_device)}
        # ControlNet/Adapter: same residuals for both batch halves (uncond and cond)
        # Multiple ControlNets: adapter_features can be a list of dicts (sum residuals)
        af = port_inputs.get("adapter_features")
        if af is not None:
            if isinstance(af, list) and af and isinstance(af[0], dict):
                # Multiple ControlNets: sum residuals element-wise (same as unet_2d_condition)
                all_down = []
                all_mid = []
                for a in af:
                    out = a.get("output") if isinstance(a.get("output"), dict) else a
                    d = out.get("down_block_additional_residuals") or out.get("down_block_residuals")
                    m = out.get("mid_block_additional_residual") or out.get("mid_block_residual")
                    if d is not None:
                        all_down.append(d if isinstance(d, (list, tuple)) else [d])
                    if m is not None:
                        all_mid.append(m)
                down_res = None
                if all_down:
                    down_res = [torch.stack(list(t)).sum(dim=0) for t in zip(*all_down)]
                mid_res = torch.stack(all_mid).sum(dim=0) if len(all_mid) > 1 else (all_mid[0] if all_mid else None)
            else:
                if isinstance(af, dict) and "output" in af and isinstance(af["output"], dict):
                    af = af["output"]
                if isinstance(af, dict):
                    down_res = af.get("down_block_additional_residuals")
                    if down_res is None:
                        down_res = af.get("down_block_residuals")
                    mid_res = af.get("mid_block_additional_residual")
                    if mid_res is None:
                        mid_res = af.get("mid_block_residual")
                elif isinstance(af, (tuple, list)) and len(af) >= 2 and not (af and isinstance(af[0], dict)):
                    down_res, mid_res = af[0], af[1]
                else:
                    down_res, mid_res = None, None
            # Ensure down_res is a sequence (list/tuple), not a single tensor (avoid bool(tensor))
            if isinstance(down_res, torch.Tensor):
                down_res = [down_res]
            if down_res is not None and isinstance(down_res, (list, tuple)) and len(down_res) > 0:
                # Duplicate for batch [uncond, cond]. UNet does sample+res so no None — use zeros if missing.
                ref_shape = None
                for r in down_res:
                    if r is not None and isinstance(r, torch.Tensor):
                        ref_shape = r.repeat(2, 1, 1, 1).to(device=model_device, dtype=model_dtype).shape
                        break
                down_list = []
                for r in down_res:
                    if r is not None and isinstance(r, torch.Tensor):
                        down_list.append(r.repeat(2, 1, 1, 1).to(device=model_device, dtype=model_dtype))
                    elif ref_shape is not None:
                        down_list.append(torch.zeros(ref_shape, device=model_device, dtype=model_dtype))
                    else:
                        down_list.append(None)
                if ref_shape is not None:
                    unet_kw["down_block_additional_residuals"] = down_list
            if mid_res is not None and isinstance(mid_res, torch.Tensor):
                mid_res_2 = mid_res.repeat(2, 1, 1, 1).to(device=model_device, dtype=model_dtype)
                unet_kw["mid_block_additional_residual"] = mid_res_2
        noise_pred = self.unet(**unet_kw)[0]
        uncond_pred, cond_pred = noise_pred.chunk(2, dim=0)
        # CFG in float32 to avoid banding from float16 cancellation (diffusers / guidance/cfg parity)
        uncond_pred = uncond_pred.float()
        cond_pred = cond_pred.float()
        guided = uncond_pred + self.scale * (cond_pred - uncond_pred)
        if self.guidance_rescale > 0.0:
            guided = _rescale_noise_cfg(guided, cond_pred, self.guidance_rescale)
        # Return float32 so solver accumulates in full precision
        return {"output": guided}
