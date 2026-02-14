from __future__ import annotations

import torch
from abc import abstractmethod
from typing import Dict, Any
from omegaconf import DictConfig

from ...core.block.base import AbstractBlock
from ...core.block.registry import register_block
from ...core.block.port import Port, InputPort, OutputPort, TensorSpec


@register_block("conditioner/abstract")
class AbstractConditioner(AbstractBlock):
    """Обработка условий (текст, ControlNet, IP-Adapter, CLAP...)."""
    
    block_type = "conditioner/abstract"
    
    @classmethod
    def declare_io(cls) -> dict:
        return {
            "raw_condition": InputPort("raw_condition", data_type="dict", description="Raw condition (text, image, audio...)"),
            "embedding": OutputPort("embedding", spec=TensorSpec(space="embedding"), description="Condition embedding"),
            "pooled_embedding": OutputPort("pooled_embedding", spec=TensorSpec(space="embedding"), description="Pooled embedding"),
            "attention_mask": OutputPort("attention_mask", description="Attention mask"),
        }
    
    def process(self, **port_inputs) -> dict:
        raw = port_inputs.get("raw_condition", port_inputs)
        if not isinstance(raw, dict):
            raw = {"text": raw}
        result = self(raw)
        
        # Normalize output keys — гарантируем что embedding всегда есть
        out = {}
        out.update(result)
        
        # Find the main embedding tensor from various key names
        emb = None
        for key in ("encoder_hidden_states", "text_emb", "embedding"):
            val = result.get(key)
            if val is not None:
                emb = val
                break
        if emb is None:
            emb = next(iter(result.values()), None)
        
        out["embedding"] = emb
        out["encoder_hidden_states"] = emb  # backbone-compatible key
        out["output"] = emb
        return out
    
    def _define_slots(self):
        return {}
    
    def _forward_impl(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Требуется AbstractBlock; делегирует в __call__(condition)."""
        condition = kwargs.get("condition", args[0] if args else {})
        return self(condition)

    @abstractmethod
    def __call__(self, condition: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """condition → dict эмбеддингов.
        
        Пример:
            {"text": "cat", "image": img_tensor} → {"text_emb": tensor, "image_emb": tensor}
        """
        pass