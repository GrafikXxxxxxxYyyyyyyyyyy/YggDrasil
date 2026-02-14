"""FreeU guidance -- backbone feature reweighting for improved quality.

Paper: "FreeU: Free Lunch in Diffusion U-Net"
"""
from __future__ import annotations

import torch
from typing import Optional, Dict, Any
from omegaconf import DictConfig

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.model.guidance import AbstractGuidance


@register_block("guidance/freeu")
class FreeUGuidance(AbstractGuidance):
    """FreeU: reweight backbone skip connections and feature maps.
    
    Enhances low-frequency components while suppressing high-frequency
    noise in UNet skip connections. Works as a post-processing hook
    on the backbone output.
    
    Parameters:
        b1, b2: Backbone feature scaling factors (default: 1.2, 1.4)
        s1, s2: Skip connection scaling factors (default: 0.9, 0.2)
    """
    
    block_type = "guidance/freeu"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.b1 = float(config.get("b1", 1.2))
        self.b2 = float(config.get("b2", 1.4))
        self.s1 = float(config.get("s1", 0.9))
        self.s2 = float(config.get("s2", 0.2))
        self.enabled = bool(config.get("enabled", True))
    
    def _forward_impl(self, *args, **kwargs) -> torch.Tensor:
        return args[0] if args else torch.zeros(1)
    
    def __call__(
        self,
        model_output: torch.Tensor,
        condition: Optional[Dict[str, Any]] = None,
        model=None,
        **kwargs,
    ) -> torch.Tensor:
        """Apply FreeU frequency filtering to model output.
        
        Enhances the backbone feature map by scaling low-frequency
        components and suppressing high-frequency noise.
        """
        if not self.enabled:
            return model_output
        
        # Apply spectral modulation if output is spatial (2D or 3D)
        if model_output.dim() >= 4:
            return self._spectral_modulate(model_output)
        
        return model_output
    
    def _spectral_modulate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frequency-domain modulation."""
        # FFT on spatial dimensions
        if x.dim() == 4:
            # 2D case (images)
            x_freq = torch.fft.fftn(x, dim=(-2, -1))
            B, C, H, W = x.shape
            
            # Create frequency mask (low-freq gets b scaling, high-freq gets s scaling)
            mask = torch.ones_like(x_freq, dtype=torch.float32)
            
            # Low-frequency region (center of FFT)
            h_center, w_center = H // 4, W // 4
            # Scale backbone features in low-freq region
            mask[:, :C//2, :h_center, :w_center] *= self.b1
            mask[:, C//2:, :h_center, :w_center] *= self.b2
            
            # Scale skip connections in high-freq region
            mask[:, :C//2, h_center:, :] *= self.s1
            mask[:, C//2:, h_center:, :] *= self.s2
            
            x_freq = x_freq * mask.to(x_freq.dtype)
            x_mod = torch.fft.ifftn(x_freq, dim=(-2, -1)).real
            
            return x_mod
        
        return x
