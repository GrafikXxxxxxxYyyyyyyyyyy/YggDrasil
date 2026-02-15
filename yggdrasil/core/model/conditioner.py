from __future__ import annotations

import torch
from typing import Dict, Any
from omegaconf import DictConfig

from ...core.block.base import AbstractBaseBlock
from ...core.block.registry import register_block
from ...core.block.port import Port, InputPort, OutputPort, TensorSpec


@register_block("conditioner/abstract")
class AbstractConditioner(AbstractBaseBlock):
    """Абстрактный conditioner (CLIP, T5, CLAP, VL, custom).
    
    Контракт: реализовать ``process()`` или ``__call__(condition)``.
    
    Порты:
        IN:  raw_condition (dict)
        OUT: embedding, pooled_embedding, attention_mask
    
    Пример::
    
        @register_block("conditioner/my_encoder")
        class MyEncoder(AbstractConditioner):
            block_type = "conditioner/my_encoder"
            def __init__(self, config):
                super().__init__(config)
                self.encoder = ...
            def process(self, **kw):
                text = kw["raw_condition"]["text"]
                emb = self.encoder(text)
                return {"embedding": emb}
    """
    
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
        
        # Try legacy __call__ for backward compat
        try:
            result = self(raw)
        except NotImplementedError:
            result = {"embedding": None}
        
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
        condition = kwargs.get("condition", args[0] if args else {})
        return self(condition)

    def __call__(self, condition: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Legacy: condition -> dict embeddings. Override process() instead."""
        raise NotImplementedError(
            f"{type(self).__name__} должен реализовать process() или __call__()"
        )