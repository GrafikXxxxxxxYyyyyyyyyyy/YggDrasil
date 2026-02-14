"""1D Transformer backbone for audio, time series, and text diffusion."""
import torch
import torch.nn as nn
import math
from typing import Optional, Dict
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.model.backbone import AbstractBackbone


class Transformer1DBlock(nn.Module):
    """Single 1D transformer block with adaLN-Zero."""
    
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=-1)
        
        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + gate1.unsqueeze(1) * h
        
        h = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate2.unsqueeze(1) * h
        
        return x


@register_block("backbone/transformer_1d")
class Transformer1DBackbone(AbstractBackbone):
    """1D Transformer backbone for sequence-based diffusion.
    
    Input: [B, C, L] (channels, length) -- e.g. audio features, time series.
    Processes as [B, L, D] token sequence through transformer blocks.
    """
    
    block_type = "backbone/transformer_1d"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        
        self.in_channels = int(config.get("in_channels", 64))
        self.out_channels = int(config.get("out_channels", self.in_channels))
        self.hidden_dim = int(config.get("hidden_dim", 512))
        self.num_layers = int(config.get("num_layers", 8))
        self.num_heads = int(config.get("num_heads", 8))
        self.mlp_ratio = float(config.get("mlp_ratio", 4.0))
        self.cond_dim = int(config.get("cond_dim", 768))
        
        # Input projection
        self.input_proj = nn.Linear(self.in_channels, self.hidden_dim)
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        
        # Conditioning projection
        self.cond_proj = nn.Linear(self.cond_dim, self.hidden_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Transformer1DBlock(self.hidden_dim, self.num_heads, self.mlp_ratio)
            for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.out_channels)
    
    def _timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.hidden_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    def _forward_impl(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        position_embedding: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # x: [B, C, L] -> [B, L, C]
        if x.dim() == 3:
            x = x.transpose(1, 2)
        elif x.dim() == 2:
            x = x.unsqueeze(-1)
        
        B, L, C = x.shape
        
        # 1. Project input
        tokens = self.input_proj(x)
        
        # 2. Position embedding
        if position_embedding is not None and position_embedding.shape[1] >= L:
            tokens = tokens + position_embedding[:, :L, :self.hidden_dim]
        
        # 3. Timestep conditioning
        t_emb = self._timestep_embedding(timestep)
        c = self.time_embed(t_emb)
        
        # 4. External conditioning
        if condition is not None:
            for key in ["encoder_hidden_states", "text_emb"]:
                if key in condition:
                    cond = condition[key]
                    if cond.dim() == 3:
                        cond = cond.mean(dim=1)
                    c = c + self.cond_proj(cond)
        
        # 5. Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, c)
        
        # 6. Output
        tokens = self.output_norm(tokens)
        output = self.output_proj(tokens)
        
        # [B, L, C_out] -> [B, C_out, L]
        output = output.transpose(1, 2)
        
        return output
