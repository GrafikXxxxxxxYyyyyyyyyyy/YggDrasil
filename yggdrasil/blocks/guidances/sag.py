"""Self-Attention Guidance (SAG).

Paper: "Improving Sample Quality of Diffusion Models Using Self-Attention Guidance"
Works in both legacy (slot-based) and graph mode.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
from omegaconf import DictConfig

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.model.guidance import AbstractGuidance


@register_block("guidance/sag")
class SelfAttentionGuidance(AbstractGuidance):
    """Self-Attention Guidance (SAG).
    
    Uses Gaussian blur on input to create a degraded prediction,
    then guides away from the degraded version.
    """
    
    block_type = "guidance/sag"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.scale = float(config.get("scale", 0.75))
        self.blur_sigma = float(config.get("blur_sigma", 2.0))
        self.threshold = float(config.get("threshold", 0.5))
        # Backbone reference for graph mode
        self._backbone_ref = None
    
    def _forward_impl(self, *args, **kwargs) -> torch.Tensor:
        return args[0] if args else torch.zeros(1)
    
    def process(self, **port_inputs) -> dict:
        """Graph-mode SAG."""
        model_output = port_inputs.get("model_output")
        
        if model_output is None or self.scale <= 0.0:
            return {"guided_output": model_output, "output": model_output}
        
        x = port_inputs.get("x")
        t = port_inputs.get("t")
        condition = port_inputs.get("condition")
        
        if self._backbone_ref is not None and x is not None and t is not None and x.dim() >= 4:
            blurred_x = self._gaussian_blur(x)
            with torch.no_grad():
                degraded_result = self._backbone_ref.process(
                    x=blurred_x, timestep=t, condition=condition,
                )
                degraded_output = degraded_result.get("output")
            
            if degraded_output is not None:
                guided = model_output + self.scale * (model_output - degraded_output)
                return {"guided_output": guided, "output": guided}
        
        return {"guided_output": model_output, "output": model_output}
    
    def __call__(
        self,
        model_output: torch.Tensor,
        condition: Optional[Dict[str, Any]] = None,
        model=None,
        x: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Legacy mode SAG."""
        if self.scale <= 0.0 or model is None or x is None or t is None:
            return model_output
        
        if x.dim() < 4:
            return model_output
        
        blurred_x = self._gaussian_blur(x)
        
        original_guidances = model._slot_children.get("guidance", [])
        model._slot_children["guidance"] = []
        
        try:
            degraded_output = model._forward_impl(
                x=blurred_x, t=t, condition=condition, return_dict=False
            )
            guided = model_output + self.scale * (model_output - degraded_output)
            return guided
        finally:
            model._slot_children["guidance"] = original_guidances
    
    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur to spatial dimensions."""
        if x.dim() == 4:
            kernel_size = int(self.blur_sigma * 4) | 1
            kernel_size = max(3, kernel_size)
            
            coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
            gauss = torch.exp(-coords ** 2 / (2 * self.blur_sigma ** 2))
            gauss = gauss / gauss.sum()
            
            kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            kernel = kernel.expand(x.shape[1], -1, -1, -1)
            
            padding = kernel_size // 2
            return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])
        
        return x
