"""Learned position embeddings."""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Optional
from omegaconf import DictConfig

from ...core.model.position import AbstractPositionEmbedder
from ...core.block.registry import register_block


@register_block("position/learned")
class LearnedEmbedder(AbstractPositionEmbedder):
    """Learned timestep and positional embeddings.
    
    Uses an nn.Embedding table for timesteps and optional
    learned spatial embeddings.
    """
    
    block_type = "position/learned"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {})
        self.dim = int(self.config.get("dim", 320))
        self.num_timesteps = int(self.config.get("num_timesteps", 1000))
        self.max_spatial = int(self.config.get("max_spatial", 4096))
        
        # Learned timestep embedding
        self.timestep_embedding = nn.Embedding(self.num_timesteps, self.dim)
        
        # Optional learned spatial embedding
        if bool(self.config.get("learn_spatial", False)):
            self.spatial_embedding = nn.Embedding(self.max_spatial, self.dim)
        else:
            self.spatial_embedding = None
        
        # MLP to project to final dim
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.SiLU(),
            nn.Linear(self.dim * 4, self.dim),
        )
    
    def _forward_impl(
        self,
        timestep: torch.Tensor,
        shape: Tuple[int, ...] = None,
    ) -> torch.Tensor:
        return self._embed(timestep, shape)
    
    def __call__(
        self,
        timestep: torch.Tensor,
        shape: Tuple[int, ...] = None,
    ) -> torch.Tensor:
        return self._embed(timestep, shape)
    
    def _embed(
        self,
        timestep: torch.Tensor,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Compute learned embedding.
        
        Args:
            timestep: [B] integer timesteps
            shape: optional (B, C, *spatial) for spatial embeddings
            
        Returns:
            [B, dim] timestep embedding
        """
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        
        # Clamp to valid range
        t_idx = timestep.long().clamp(0, self.num_timesteps - 1)
        t_emb = self.timestep_embedding(t_idx)
        t_emb = self.mlp(t_emb)
        
        return t_emb
