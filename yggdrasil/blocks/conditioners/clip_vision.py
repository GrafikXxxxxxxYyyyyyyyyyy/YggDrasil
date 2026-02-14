"""CLIP Vision conditioner — encodes images to CLIP embeddings.

Used by image-conditioned pipelines (I2V, image variation, etc.)
"""
import torch
import torch.nn as nn
from typing import Dict, Any
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.block.port import InputPort, OutputPort, TensorSpec
from ...core.model.conditioner import AbstractConditioner


@register_block("conditioner/clip_vision")
class CLIPVisionConditioner(AbstractConditioner):
    """CLIP vision encoder — image -> embedding."""
    
    block_type = "conditioner/clip_vision"
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "conditioner/clip_vision"}
        super().__init__(config)
        self._model = None
        self._processor = None
        self.pretrained = self.config.get("pretrained", "openai/clip-vit-large-patch14")
        self._build_model()
    
    @classmethod
    def declare_io(cls) -> dict:
        return {
            "raw_condition": InputPort("raw_condition", data_type="dict",
                                       description="Dict with 'image' key (PIL or tensor)"),
            "embedding": OutputPort("embedding", spec=TensorSpec(space="embedding")),
            "pooled_embedding": OutputPort("pooled_embedding", spec=TensorSpec(space="embedding")),
        }
    
    def _build_model(self):
        try:
            from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
            self._model = CLIPVisionModelWithProjection.from_pretrained(self.pretrained)
            self._processor = CLIPImageProcessor.from_pretrained(self.pretrained)
            self._model.requires_grad_(False)
        except Exception:
            self._model = None
    
    def process(self, **port_inputs) -> dict:
        raw = port_inputs.get("raw_condition", {})
        if not isinstance(raw, dict):
            raw = {"image": raw}
        
        image = raw.get("image")
        if image is None:
            emb = torch.zeros(1, 1, 768)
            return {"embedding": emb, "pooled_embedding": emb[:, 0], "output": emb}
        
        if self._model is not None and self._processor is not None:
            with torch.no_grad():
                inputs = self._processor(images=image, return_tensors="pt")
                inputs = {k: v.to(next(self._model.parameters()).device) for k, v in inputs.items()}
                outputs = self._model(**inputs)
                emb = outputs.image_embeds.unsqueeze(1)
                return {
                    "embedding": emb,
                    "pooled_embedding": outputs.image_embeds,
                    "output": emb,
                }
        
        # Stub
        emb = torch.randn(1, 1, 768)
        return {"embedding": emb, "pooled_embedding": emb[:, 0], "output": emb}
    
    def __call__(self, condition):
        return self.process(raw_condition=condition)
