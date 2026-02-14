# yggdrasil/blocks/guidances/cfg.py
"""Classifier-Free Guidance — port-based, with optional guidance_rescale (diffusers parity).

Формула: guided = uncond + scale * (cond - uncond)
+ optional rescale_noise_cfg (Section 3.4, Common Diffusion Noise Schedules...).
"""
from __future__ import annotations

import torch
from omegaconf import DictConfig
from typing import Optional, Dict, Any

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import InputPort, OutputPort, TensorSpec
from yggdrasil.core.model.guidance import AbstractGuidance


def _rescale_noise_cfg(noise_cfg: torch.Tensor, noise_pred_text: torch.Tensor, guidance_rescale: float) -> torch.Tensor:
    """Rescale CFG noise to fix overexposure (diffusers parity)."""
    if guidance_rescale <= 0.0:
        return noise_cfg
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True).clamp(min=1e-8)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True).clamp(min=1e-8)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    return guidance_rescale * noise_pred_rescaled + (1.0 - guidance_rescale) * noise_cfg


@register_block("guidance/cfg")
class ClassifierFreeGuidance(AbstractGuidance):
    """Classifier-Free Guidance.
    
    Config: scale (7.5), guidance_rescale (0.0 = off; 0.7 often better for SD 1.5).
    """
    
    block_type = "guidance/cfg"
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "guidance/cfg"}
        super().__init__(config)
        self.scale = float(self.config.get("scale", 7.5))
        self.guidance_rescale = float(self.config.get("guidance_rescale", 0.0))
    
    @classmethod
    def declare_io(cls) -> dict:
        return {
            "model_output": InputPort("model_output", spec=TensorSpec(space="latent"),
                                      description="Conditional model prediction"),
            "uncond_output": InputPort("uncond_output", spec=TensorSpec(space="latent"),
                                       optional=True,
                                       description="Unconditional model prediction"),
            "guided_output": OutputPort("guided_output", spec=TensorSpec(space="latent"),
                                         description="Guided output"),
        }
    
    def process(self, **port_inputs) -> dict:
        model_output = port_inputs.get("model_output")
        uncond_output = port_inputs.get("uncond_output")
        
        if model_output is None or self.scale <= 1.0 or uncond_output is None:
            return {"guided_output": model_output, "output": model_output}
        
        # Compute in float32 to avoid banding from float16 cancellation in (cond - uncond)
        model_output = model_output.float()
        uncond_output = uncond_output.float()
        guided = uncond_output + self.scale * (model_output - uncond_output)
        if self.guidance_rescale > 0.0:
            guided = _rescale_noise_cfg(guided, model_output, self.guidance_rescale)
        return {"guided_output": guided, "output": guided}
    
    def _forward_impl(self, *args, **kwargs) -> torch.Tensor:
        return args[0] if args else torch.zeros(1)
    
    def __call__(self, model_output, **kwargs) -> torch.Tensor:
        """Legacy compat."""
        uncond = kwargs.get("uncond_output")
        if uncond is not None and self.scale > 1.0:
            return uncond + self.scale * (model_output - uncond)
        return model_output
