"""MMDiT (Multimodal Diffusion Transformer) backbone.

Used by SD3 and Flux. Features joint attention between
image tokens and text tokens.
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Dict
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.model.backbone import AbstractBackbone


class JointAttentionBlock(nn.Module):
    """Joint attention block for MMDiT.
    
    Processes image and text tokens in a shared attention space.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Image stream
        self.norm1_img = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.qkv_img = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj_img = nn.Linear(hidden_dim, hidden_dim)
        self.norm2_img = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp_img = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
        
        # Text stream
        self.norm1_txt = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.qkv_txt = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj_txt = nn.Linear(hidden_dim, hidden_dim)
        self.norm2_txt = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp_txt = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
        
        # adaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 12 * hidden_dim),
        )
    
    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple:
        """
        Args:
            img: [B, S_img, D] image tokens
            txt: [B, S_txt, D] text tokens  
            c: [B, D] conditioning
        """
        mod = self.adaLN_modulation(c).chunk(12, dim=-1)
        (shift_img_attn, scale_img_attn, gate_img_attn,
         shift_img_mlp, scale_img_mlp, gate_img_mlp,
         shift_txt_attn, scale_txt_attn, gate_txt_attn,
         shift_txt_mlp, scale_txt_mlp, gate_txt_mlp) = mod
        
        B = img.shape[0]
        
        # Pre-norm with adaLN
        img_norm = self.norm1_img(img) * (1 + scale_img_attn.unsqueeze(1)) + shift_img_attn.unsqueeze(1)
        txt_norm = self.norm1_txt(txt) * (1 + scale_txt_attn.unsqueeze(1)) + shift_txt_attn.unsqueeze(1)
        
        # QKV projections
        q_img, k_img, v_img = self.qkv_img(img_norm).chunk(3, dim=-1)
        q_txt, k_txt, v_txt = self.qkv_txt(txt_norm).chunk(3, dim=-1)
        
        # Concatenate for joint attention
        q = torch.cat([q_img, q_txt], dim=1)
        k = torch.cat([k_img, k_txt], dim=1)
        v = torch.cat([v_img, v_txt], dim=1)
        
        # Reshape for multi-head attention
        S = q.shape[1]
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, S, -1)
        
        # Split back
        S_img = img.shape[1]
        attn_img = self.proj_img(attn[:, :S_img])
        attn_txt = self.proj_txt(attn[:, S_img:])
        
        # Residual + gate
        img = img + gate_img_attn.unsqueeze(1) * attn_img
        txt = txt + gate_txt_attn.unsqueeze(1) * attn_txt
        
        # MLP
        img_mlp = self.norm2_img(img) * (1 + scale_img_mlp.unsqueeze(1)) + shift_img_mlp.unsqueeze(1)
        img = img + gate_img_mlp.unsqueeze(1) * self.mlp_img(img_mlp)
        
        txt_mlp = self.norm2_txt(txt) * (1 + scale_txt_mlp.unsqueeze(1)) + shift_txt_mlp.unsqueeze(1)
        txt = txt + gate_txt_mlp.unsqueeze(1) * self.mlp_txt(txt_mlp)
        
        return img, txt


@register_block("backbone/mmdit")
class MMDiTBackbone(AbstractBackbone):
    """Multimodal Diffusion Transformer (MMDiT) backbone.
    
    Features joint attention between image and text tokens.
    Used by SD3 and Flux architectures.
    """
    
    block_type = "backbone/mmdit"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        
        self.hidden_dim = int(config.get("hidden_dim", 1536))
        self.num_layers = int(config.get("num_layers", 24))
        self.num_heads = int(config.get("num_heads", 24))
        self.patch_size = int(config.get("patch_size", 2))
        self.in_channels = int(config.get("in_channels", 16))
        self.out_channels = int(config.get("out_channels", self.in_channels))
        self.mlp_ratio = float(config.get("mlp_ratio", 4.0))
        self.cond_dim = int(config.get("cond_dim", 4096))
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            self.in_channels, self.hidden_dim,
            kernel_size=self.patch_size, stride=self.patch_size,
        )
        
        # Text projection
        self.text_proj = nn.Linear(self.cond_dim, self.hidden_dim)
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        
        # Joint attention blocks
        self.blocks = nn.ModuleList([
            JointAttentionBlock(self.hidden_dim, self.num_heads, self.mlp_ratio)
            for _ in range(self.num_layers)
        ])
        
        # Final layer
        self.final_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
        self.final_linear = nn.Linear(
            self.hidden_dim, self.patch_size ** 2 * self.out_channels
        )
        
        self._init_weights()
    
    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)
    
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
        B, C, H, W = x.shape
        ph, pw = H // self.patch_size, W // self.patch_size
        
        # 1. Patchify image
        img_tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # 2. Position embedding
        if position_embedding is not None and position_embedding.shape[1] >= img_tokens.shape[1]:
            img_tokens = img_tokens + position_embedding[:, :img_tokens.shape[1], :self.hidden_dim]
        
        # 3. Text tokens
        txt_tokens = torch.zeros(B, 1, self.hidden_dim, device=x.device, dtype=x.dtype)
        if condition is not None:
            for key in ["encoder_hidden_states", "text_emb"]:
                if key in condition:
                    txt_tokens = self.text_proj(condition[key])
                    break
        
        # 4. Timestep conditioning
        t_emb = self._timestep_embedding(timestep)
        c = self.time_embed(t_emb)
        
        # 5. Joint attention blocks
        for block in self.blocks:
            img_tokens, txt_tokens = block(img_tokens, txt_tokens, c)
        
        # 6. Final norm + linear (image only)
        img_tokens = self.final_norm(img_tokens)
        img_tokens = self.final_linear(img_tokens)
        
        # 7. Unpatchify
        p = self.patch_size
        output = img_tokens.reshape(B, ph, pw, p, p, self.out_channels)
        output = output.permute(0, 5, 1, 3, 2, 4).reshape(B, self.out_channels, H, W)
        
        return output
