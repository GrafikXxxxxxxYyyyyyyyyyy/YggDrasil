"""NullConditioner — outputs zero embeddings of a given shape.

Used in CFG graphs as the unconditional branch:
    conditioner --> backbone_cond  --|
                                    |--> guidance --> solver
    null_cond   --> backbone_uncond -|
"""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Dict, Any

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import InputPort, OutputPort, TensorSpec
from yggdrasil.core.model.conditioner import AbstractConditioner


@register_block("conditioner/null")
class NullConditioner(AbstractConditioner):
    """Outputs zero embeddings matching a reference conditioner's shape.
    
    Uses a registered buffer to track device/dtype correctly when
    the graph is moved via .to(device).
    
    Config:
        embedding_dim: int (default 768) — embedding dimension
        seq_length: int (default 77) — sequence length
    """
    
    block_type = "conditioner/null"
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "conditioner/null"}
        super().__init__(config)
        self.embedding_dim = int(self.config.get("embedding_dim", 768))
        self.seq_length = int(self.config.get("seq_length", 77))
        
        # Register a buffer so .to(device, dtype) propagates correctly
        self.register_buffer(
            "_device_tracker",
            torch.zeros(1),
        )
    
    def process(self, **port_inputs) -> dict:
        raw = port_inputs.get("raw_condition", {})
        batch_size = 1
        
        # Infer batch_size from any incoming tensor
        for v in (port_inputs.get("reference_embedding"), raw):
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                batch_size = v.shape[0]
                break
            if isinstance(v, dict):
                for tv in v.values():
                    if isinstance(tv, torch.Tensor) and tv.dim() >= 1:
                        batch_size = tv.shape[0]
                        break
        
        # Use the registered buffer to get correct device and dtype
        device = self._device_tracker.device
        dtype = self._device_tracker.dtype
        
        # Try to infer shape from reference embedding
        ref = port_inputs.get("reference_embedding")
        if ref is not None and isinstance(ref, torch.Tensor):
            emb = torch.zeros_like(ref)
        else:
            emb = torch.zeros(
                batch_size, self.seq_length, self.embedding_dim,
                device=device, dtype=dtype,
            )
        
        return {
            "embedding": emb,
            "encoder_hidden_states": emb,
            "output": emb,
        }
    
    def __call__(self, condition: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        device = self._device_tracker.device
        dtype = self._device_tracker.dtype
        emb = torch.zeros(1, self.seq_length, self.embedding_dim, device=device, dtype=dtype)
        return {"encoder_hidden_states": emb, "embedding": emb}
