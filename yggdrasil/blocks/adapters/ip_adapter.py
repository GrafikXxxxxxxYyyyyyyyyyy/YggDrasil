# yggdrasil/blocks/adapters/ip_adapter.py
"""IP-Adapter — image prompt adapter for diffusion models.

Adds image-conditioned generation by injecting image features into
cross-attention layers. Works as a Lego block: can be injected into any
backbone that uses cross-attention.

Usage::

    ip_adapter = IPAdapter({"type": "adapter/ip_adapter", "scale": 0.6})
    ip_adapter.inject_into(backbone_block)
    
    # In graph
    graph.add_node("ip_adapter", ip_adapter)
    graph.connect("image_encoder", "features", "ip_adapter", "image_features")
"""
from __future__ import annotations

import logging
import math
import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import InputPort, OutputPort
from .base import AbstractAdapter

logger = logging.getLogger(__name__)


class ImageProjection(nn.Module):
    """Projects image features to cross-attention dimension.
    
    Takes CLIP image features and projects them to the same dimension
    as text embeddings for cross-attention injection.
    """
    
    def __init__(
        self,
        image_embed_dim: int = 1024,
        cross_attention_dim: int = 768,
        num_tokens: int = 4,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        
        self.proj = nn.Linear(image_embed_dim, cross_attention_dim * num_tokens)
        self.norm = nn.LayerNorm(cross_attention_dim)
    
    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """Project image embeddings to cross-attention tokens.
        
        Args:
            image_embeds: (batch, image_embed_dim)
            
        Returns:
            (batch, num_tokens, cross_attention_dim)
        """
        x = self.proj(image_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x


class IPAttnProcessor(nn.Module):
    """Attention processor that adds IP-Adapter image features.
    
    Injects image prompt tokens into the cross-attention layer by:
    1. Computing normal text cross-attention
    2. Computing image cross-attention with separate K/V projections
    3. Adding them with a scale factor
    """
    
    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        
        # Separate K/V projections for image features
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
    
    def forward(
        self,
        attn_output: torch.Tensor,
        hidden_states: torch.Tensor,
        ip_hidden_states: torch.Tensor,
        num_heads: int,
    ) -> torch.Tensor:
        """Add IP-Adapter attention to existing attention output.
        
        Args:
            attn_output: Output from normal cross-attention (batch, seq, hidden)
            hidden_states: Query states (batch, seq, hidden)
            ip_hidden_states: Image prompt tokens (batch, num_tokens, cross_attn_dim)
            num_heads: Number of attention heads
            
        Returns:
            Modified attention output
        """
        batch_size = hidden_states.shape[0]
        head_dim = self.hidden_size // num_heads
        
        # Project image features
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)
        
        # Reshape for multi-head attention
        ip_key = ip_key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        
        # Query from hidden states (reuse from main attention)
        # For simplicity, use a scaled dot-product attention
        query = hidden_states.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        
        # Compute attention
        scale_factor = 1.0 / math.sqrt(head_dim)
        attn_weights = torch.matmul(query, ip_key.transpose(-2, -1)) * scale_factor
        attn_weights = torch.softmax(attn_weights, dim=-1)
        ip_output = torch.matmul(attn_weights, ip_value)
        
        # Reshape back
        ip_output = ip_output.transpose(1, 2).reshape(batch_size, -1, self.hidden_size)
        
        # Add to original attention with scale
        return attn_output + self.scale * ip_output


@register_block("adapter/ip_adapter")
class IPAdapter(AbstractAdapter):
    """IP-Adapter — image prompt for cross-attention injection.
    
    Features:
    - Image projection (CLIP -> cross-attention dim)
    - Per-layer attention processors
    - Configurable scale
    - Trainable: can fine-tune the projector and attention processors
    - Save/load weights
    
    Example::
    
        adapter = IPAdapter({
            "type": "adapter/ip_adapter",
            "image_embed_dim": 1024,
            "cross_attention_dim": 768,
            "scale": 0.6,
            "num_tokens": 4,
        })
        adapter.inject_into(backbone)
    """
    
    block_type = "adapter/ip_adapter"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.scale = config.get("scale", 1.0)
        self.image_embed_dim = config.get("image_embed_dim", 1024)
        self.cross_attention_dim = config.get("cross_attention_dim", 768)
        self.num_tokens = config.get("num_tokens", 4)
        
        # Image projection layer
        self.image_proj = ImageProjection(
            image_embed_dim=self.image_embed_dim,
            cross_attention_dim=self.cross_attention_dim,
            num_tokens=self.num_tokens,
        )
        
        # Attention processors (created during inject_into)
        self.ip_attn_processors: nn.ModuleDict = nn.ModuleDict()
        self._injected = False
    
    @classmethod
    def declare_io(cls):
        return {
            "image_features": InputPort("image_features", 
                                         description="CLIP image embeddings (batch, embed_dim)"),
            "image_prompt_embeds": OutputPort("image_prompt_embeds",
                                              description="Projected tokens for cross-attention"),
        }
    
    def process(self, **kw) -> Dict[str, Any]:
        """Project image features to cross-attention tokens."""
        image_features = kw.get("image_features")
        if image_features is None:
            return {"image_prompt_embeds": None}
        
        # Project to cross-attention space
        image_prompt_embeds = self.image_proj(image_features)
        return {"image_prompt_embeds": image_prompt_embeds}
    
    def inject_into(self, target):
        """Inject IP-Adapter attention processors into target backbone.
        
        Scans for cross-attention layers (attn2) and adds IP attention processors.
        """
        injected_count = 0
        
        for name, module in target.named_modules():
            # Look for cross-attention layers
            if hasattr(module, "to_k") and hasattr(module, "to_v"):
                # Determine hidden size from existing projections
                if hasattr(module.to_k, "weight"):
                    hidden_size = module.to_k.weight.shape[0]
                elif hasattr(module.to_k, "out_features"):
                    hidden_size = module.to_k.out_features
                else:
                    continue
                
                processor = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=self.cross_attention_dim,
                    scale=self.scale,
                )
                safe_name = name.replace(".", "_")
                self.ip_attn_processors[safe_name] = processor
                
                # Store reference on the module for the hook
                module._ip_processor = processor
                injected_count += 1
        
        self._injected = True
        logger.info(f"IP-Adapter injected into {injected_count} attention layers")
    
    def apply(self, output: torch.Tensor, context=None) -> torch.Tensor:
        """Apply IP-Adapter to attention output (used in hook mode)."""
        return output
    
    def set_scale(self, scale: float):
        """Update scale for all attention processors."""
        self.scale = scale
        for proc in self.ip_attn_processors.values():
            proc.scale = scale
    
    def save_weights(self, path: str):
        """Save only IP-Adapter weights (projection + attention processors)."""
        state = {
            "image_proj": self.image_proj.state_dict(),
            "ip_attn_processors": self.ip_attn_processors.state_dict(),
        }
        torch.save(state, path)
        logger.info(f"Saved IP-Adapter weights to {path}")
    
    def load_weights(self, path: str):
        """Load IP-Adapter weights."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.image_proj.load_state_dict(state["image_proj"])
        if "ip_attn_processors" in state:
            self.ip_attn_processors.load_state_dict(state["ip_attn_processors"])
        logger.info(f"Loaded IP-Adapter weights from {path}")
