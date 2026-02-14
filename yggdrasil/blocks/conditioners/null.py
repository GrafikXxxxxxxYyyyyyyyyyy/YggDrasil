"""NullConditioner — выдаёт нулевые эмбеддинги заданного размера.

Используется в CFG-графах как unconditional branch:
    conditioner --> backbone_cond  --|
                                    |--> guidance --> solver
    null_cond   --> backbone_uncond -|
"""
from __future__ import annotations

import torch
from omegaconf import DictConfig
from typing import Dict, Any

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import InputPort, OutputPort, TensorSpec
from yggdrasil.core.model.conditioner import AbstractConditioner


@register_block("conditioner/null")
class NullConditioner(AbstractConditioner):
    """Outputs zero embeddings matching a reference conditioner's shape.
    
    Config:
        embedding_dim: int (default 768) — embedding dimension
        seq_length: int (default 77) — sequence length
        batch_size: int (default 1)
    """
    
    block_type = "conditioner/null"
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "conditioner/null"}
        super().__init__(config)
        self.embedding_dim = int(self.config.get("embedding_dim", 768))
        self.seq_length = int(self.config.get("seq_length", 77))
    
    def process(self, **port_inputs) -> dict:
        raw = port_inputs.get("raw_condition", {})
        batch_size = 1
        
        # Try to infer shape from reference embedding
        ref = port_inputs.get("reference_embedding")
        if ref is not None and isinstance(ref, torch.Tensor):
            emb = torch.zeros_like(ref)
        else:
            device = torch.device("cpu")
            # Try to get device from raw condition
            if isinstance(raw, dict):
                for v in raw.values():
                    if isinstance(v, torch.Tensor):
                        device = v.device
                        break
            emb = torch.zeros(batch_size, self.seq_length, self.embedding_dim, device=device)
        
        return {
            "embedding": emb,
            "encoder_hidden_states": emb,
            "output": emb,
        }
    
    def __call__(self, condition: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        emb = torch.zeros(1, self.seq_length, self.embedding_dim)
        return {"encoder_hidden_states": emb, "embedding": emb}
