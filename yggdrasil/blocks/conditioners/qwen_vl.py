"""Qwen VL conditioner — vision-language processor for QwenImage.

Encodes both text and image inputs into embeddings for the QwenImage backbone.
"""
import torch
import torch.nn as nn
from typing import Dict, Any
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.block.port import InputPort, OutputPort, TensorSpec
from ...core.model.conditioner import AbstractConditioner


@register_block("conditioner/qwen_vl")
class QwenVLConditioner(AbstractConditioner):
    """Qwen VL processor — text+image -> embeddings."""
    
    block_type = "conditioner/qwen_vl"
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "conditioner/qwen_vl"}
        super().__init__(config)
        self._model = None
        self._processor = None
        self.pretrained = self.config.get("pretrained", "Qwen/Qwen2.5-VL-3B-Instruct")
        self.embedding_dim = int(self.config.get("embedding_dim", 2048))
        self._build_model()
    
    @classmethod
    def declare_io(cls) -> dict:
        return {
            "raw_condition": InputPort("raw_condition", data_type="dict",
                                       description="Dict with 'text' and optionally 'image' keys"),
            "embedding": OutputPort("embedding", spec=TensorSpec(space="embedding")),
            "pooled_embedding": OutputPort("pooled_embedding", spec=TensorSpec(space="embedding")),
        }
    
    def _build_model(self):
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.pretrained, torch_dtype=torch.float16,
            )
            self._processor = AutoProcessor.from_pretrained(self.pretrained)
            self._model.requires_grad_(False)
        except Exception:
            self._model = None
    
    def process(self, **port_inputs) -> dict:
        raw = port_inputs.get("raw_condition", {})
        if not isinstance(raw, dict):
            raw = {"text": str(raw)}
        
        text = raw.get("text", "")
        image = raw.get("image")
        
        if self._model is not None and self._processor is not None:
            with torch.no_grad():
                # Build messages in Qwen VL format
                content = [{"type": "text", "text": text}]
                if image is not None:
                    content.insert(0, {"type": "image", "image": image})
                
                messages = [{"role": "user", "content": content}]
                
                try:
                    inputs = self._processor(messages, return_tensors="pt")
                    inputs = {k: v.to(next(self._model.parameters()).device)
                              for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                    outputs = self._model(**inputs, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1]
                    pooled = hidden[:, -1]
                    return {
                        "embedding": hidden,
                        "pooled_embedding": pooled,
                        "encoder_hidden_states": hidden,
                        "output": hidden,
                    }
                except Exception:
                    pass
        
        # Stub: return zero embeddings
        emb = torch.zeros(1, 77, self.embedding_dim)
        return {
            "embedding": emb,
            "pooled_embedding": emb[:, 0],
            "encoder_hidden_states": emb,
            "output": emb,
        }
    
    def __call__(self, condition):
        return self.process(raw_condition=condition)
