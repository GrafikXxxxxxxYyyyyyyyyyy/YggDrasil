"""N-dimensional Rotary Position Embeddings (RoPE-ND).

Used by DiT, Flux, SD3 and other transformer-based diffusion models.
Supports arbitrary spatial dimensions (1D for audio, 2D for images, 3D for video).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional
from omegaconf import DictConfig

from ...core.model.position import AbstractPositionEmbedder
from ...core.block.registry import register_block


@register_block("position/rope_nd")
class RoPENDEmbedder(AbstractPositionEmbedder):
    """N-dimensional RoPE for transformer-based diffusion backbones.
    
    Generates rotary position embeddings that can handle
    any number of spatial dimensions.
    """
    
    block_type = "position/rope_nd"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {})
        self.dim = int(self.config.get("dim", 64))
        self.theta = float(self.config.get("theta", 10000.0))
        self.max_seq_len = int(self.config.get("max_seq_len", 8192))
        self.include_timestep = bool(self.config.get("include_timestep", True))
    
    def _forward_impl(
        self,
        timestep: torch.Tensor,
        shape: Tuple[int, ...] = None,
    ) -> torch.Tensor:
        return self._compute_rope(timestep, shape)
    
    def __call__(
        self,
        timestep: torch.Tensor,
        shape: Tuple[int, ...] = None,
    ) -> torch.Tensor:
        return self._compute_rope(timestep, shape)
    
    def _compute_rope(
        self,
        timestep: torch.Tensor,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Compute n-dimensional RoPE embeddings.
        
        Args:
            timestep: [B] timestep tensor
            shape: (B, C, *spatial_dims) tensor shape for spatial RoPE
            
        Returns:
            Position embedding tensor suitable for the backbone.
        """
        device = timestep.device
        
        if shape is None:
            # Timestep-only embedding
            return self._timestep_rope(timestep)
        
        spatial_dims = shape[2:]  # e.g. (H, W) or (T, H, W) or (L,)
        ndim = len(spatial_dims)
        
        if ndim == 0:
            return self._timestep_rope(timestep)
        
        # Split embedding dim across spatial dimensions
        dim_per_axis = self.dim // ndim
        
        all_freqs = []
        for axis_idx, axis_len in enumerate(spatial_dims):
            # Frequencies for this axis
            half_dim = dim_per_axis // 2
            freqs = 1.0 / (
                self.theta ** (torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim)
            )
            positions = torch.arange(axis_len, dtype=torch.float32, device=device)
            
            # Outer product: [axis_len, half_dim]
            angles = torch.outer(positions, freqs)
            # Complex exponential for RoPE
            cos_angles = torch.cos(angles)
            sin_angles = torch.sin(angles)
            
            all_freqs.append(torch.stack([cos_angles, sin_angles], dim=-1))
        
        # Combine all axes into a single embedding
        # For simplicity, we concatenate the axis embeddings
        # Each element: [axis_len, half_dim, 2]
        # We need to create a meshgrid and flatten
        if ndim == 1:
            rope = all_freqs[0].reshape(-1, dim_per_axis)
        elif ndim == 2:
            h, w = spatial_dims
            rope_h = all_freqs[0]  # [H, half_dim, 2]
            rope_w = all_freqs[1]  # [W, half_dim, 2]
            # Broadcast to [H, W, dim_per_axis * 2]
            rope_h = rope_h.unsqueeze(1).expand(-1, w, -1, -1).reshape(h * w, dim_per_axis)
            rope_w = rope_w.unsqueeze(0).expand(h, -1, -1, -1).reshape(h * w, dim_per_axis)
            rope = torch.cat([rope_h, rope_w], dim=-1)
        elif ndim == 3:
            t_len, h, w = spatial_dims
            rope_t = all_freqs[0].unsqueeze(1).unsqueeze(1).expand(-1, h, w, -1, -1)
            rope_h = all_freqs[1].unsqueeze(0).unsqueeze(2).expand(t_len, -1, w, -1, -1)
            rope_w = all_freqs[2].unsqueeze(0).unsqueeze(0).expand(t_len, h, -1, -1, -1)
            rope_t = rope_t.reshape(-1, dim_per_axis)
            rope_h = rope_h.reshape(-1, dim_per_axis)
            rope_w = rope_w.reshape(-1, dim_per_axis)
            rope = torch.cat([rope_t, rope_h, rope_w], dim=-1)
        else:
            # Fallback for higher dims: just concatenate
            rope = torch.cat([f.reshape(-1, dim_per_axis) for f in all_freqs], dim=-1)
        
        # Expand for batch: [1, seq_len, dim] -> [B, seq_len, dim]
        batch_size = shape[0]
        rope = rope.unsqueeze(0).expand(batch_size, -1, -1)
        
        return rope
    
    def _timestep_rope(self, timestep: torch.Tensor) -> torch.Tensor:
        """Simple timestep-only RoPE embedding."""
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        
        half = self.dim // 2
        freqs = 1.0 / (
            self.theta ** (torch.arange(0, half, dtype=torch.float32, device=timestep.device) / half)
        )
        angles = timestep.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
