# yggdrasil/blocks/adapters/fusion.py
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Dict, Any, List

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.model.backbone import AbstractBackbone
from .base import AbstractAdapter


@register_block("adapter/fusion/cross_attention")
class CrossAttentionFusionAdapter(AbstractAdapter):
    """Cross-Attention Fusion — мощный мультимодальный адаптер."""
    
    block_type = "adapter/fusion/cross_attention"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.modalities = config.get("modalities", ["text", "image"])
        self.dim = config.get("dim", 768)
        self.num_heads = config.get("num_heads", 8)
        self.scale = config.get("scale", 1.0)
        self.projectors = nn.ModuleDict({
            mod: nn.Linear(self.dim, self.dim) for mod in self.modalities
        })
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.num_heads,
            batch_first=True
        )

    def inject_into(self, target: AbstractBackbone):
        if hasattr(target, "unet"):
            original_forward = target.unet.forward
            
            def fused_forward(*args, **kwargs):
                condition = kwargs.get("condition", {})
                fused = self._fuse_conditions(condition)
                condition.update(fused)
                return original_forward(*args, **kwargs)
            
            target.unet.forward = fused_forward
        else:
            target.add_pre_hook(self._pre_forward_hook)
    
    def _pre_forward_hook(self, module, x, timestep, condition: Dict[str, Any], **kwargs):
        """Слияние условий."""
        fused = self._fuse_conditions(condition)
        condition.update(fused)
        return x, timestep, condition   # ← ИСПРАВЛЕНО
    
    def _fuse_conditions(self, condition: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        fused_embs = []
        for mod_name in self.modalities:
            if mod_name in condition and isinstance(condition[mod_name], torch.Tensor):
                proj = self.projectors[mod_name](condition[mod_name])
                fused_embs.append(proj)
        
        if not fused_embs:
            return {}
        
        fused = torch.cat(fused_embs, dim=1)
        fused, _ = self.cross_attn(fused, fused, fused)
        return {"fused_condition": fused * self.scale}
    
    def apply(self, output: torch.Tensor, context=None) -> torch.Tensor:
        return output


@register_block("adapter/fusion/late")
class LateFusionAdapter(AbstractAdapter):
    """Простой late fusion."""
    
    block_type = "adapter/fusion/late"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.fusion_mode = config.get("mode", "sum")
        self.scale = config.get("scale", 1.0)
    
    def _pre_forward_hook(self, module, x, timestep, condition: Dict[str, Any], **kwargs):
        fused = self._fuse_conditions(condition)
        condition.update(fused)
        return x, timestep, condition
    
    def _fuse_conditions(self, condition: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        embs = [v for v in condition.values() if isinstance(v, torch.Tensor) and v.ndim >= 2]
        if not embs:
            return {}
        if self.fusion_mode == "sum":
            fused = sum(embs) * self.scale
        elif self.fusion_mode == "mean":
            fused = torch.mean(torch.stack(embs), dim=0) * self.scale
        else:
            fused = torch.cat(embs, dim=-1) * self.scale
        return {"fused_condition": fused}