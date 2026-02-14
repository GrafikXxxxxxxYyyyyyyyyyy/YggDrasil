"""DiT (Diffusion Transformer) backbone.

Used by SD3, Lumina, and other transformer-based diffusion models.
Processes latents as sequences of patches.
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Dict
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.model.backbone import AbstractBackbone


class DiTBlock(nn.Module):
    """Single DiT transformer block with adaptive layer norm (adaLN-Zero)."""
    
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
        # adaLN-Zero: 6 modulation parameters (shift, scale for norm1, norm2, + gate for attn, mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D] sequence of patch embeddings
            c: [B, D] conditioning (timestep + class/text embedding)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # Self-attention with adaLN
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + gate_msa.unsqueeze(1) * h
        
        # MLP with adaLN
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h
        
        return x


@register_block("backbone/dit")
class DiTBackbone(AbstractBackbone):
    """Diffusion Transformer backbone.
    
    Patchifies input latents, processes through transformer blocks
    with adaLN-Zero conditioning, then unpatchifies.
    
    Supports loading pretrained weights from diffusers or custom checkpoints.
    """
    
    block_type = "backbone/dit"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        
        self.hidden_dim = int(config.get("hidden_dim", 1152))
        self.num_layers = int(config.get("num_layers", 28))
        self.num_heads = int(config.get("num_heads", 16))
        self.patch_size = int(config.get("patch_size", 2))
        self.in_channels = int(config.get("in_channels", 4))
        self.out_channels = int(config.get("out_channels", self.in_channels))
        self.mlp_ratio = float(config.get("mlp_ratio", 4.0))
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            self.in_channels, self.hidden_dim,
            kernel_size=self.patch_size, stride=self.patch_size,
        )
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        
        # Optional text/class conditioning projection
        self.cond_proj = nn.Linear(
            int(config.get("cond_dim", 768)), self.hidden_dim
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_dim, self.num_heads, self.mlp_ratio)
            for _ in range(self.num_layers)
        ])
        
        # Final norm and linear
        self.final_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
        )
        self.final_linear = nn.Linear(
            self.hidden_dim, self.patch_size ** 2 * self.out_channels
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)
        # Zero-init the final linear for stable training
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)
    
    def _timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal timestep embedding."""
        half = self.hidden_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, S, D] where S = (H/p) * (W/p)."""
        return self.patch_embed(x).flatten(2).transpose(1, 2)
    
    def unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """[B, S, p*p*C] -> [B, C, H, W]."""
        p = self.patch_size
        c = self.out_channels
        x = x.reshape(-1, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(-1, c, h * p, w * p)
        return x
    
    def _forward_impl(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        position_embedding: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        ph, pw = H // self.patch_size, W // self.patch_size
        
        # 1. Patchify
        tokens = self.patchify(x)
        
        # 2. Add position embedding if provided
        if position_embedding is not None:
            if position_embedding.shape[1] == tokens.shape[1]:
                tokens = tokens + position_embedding[:, :tokens.shape[1], :self.hidden_dim]
        
        # 3. Timestep conditioning
        t_emb = self._timestep_embedding(timestep)
        t_emb = self.time_embed(t_emb)
        
        # 4. Text/class conditioning
        cond_emb = torch.zeros_like(t_emb)
        if condition is not None:
            for key in ["encoder_hidden_states", "text_emb", "class_emb"]:
                if key in condition:
                    c = condition[key]
                    if c.dim() == 3:
                        c = c.mean(dim=1)  # Pool sequence dim
                    cond_emb = cond_emb + self.cond_proj(c)
        
        c = t_emb + cond_emb
        
        # 5. Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, c)
        
        # 6. Final layer
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        tokens = self.final_norm(tokens) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        tokens = self.final_linear(tokens)
        
        # 7. Unpatchify
        output = self.unpatchify(tokens, ph, pw)
        
        return output
