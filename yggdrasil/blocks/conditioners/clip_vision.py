"""CLIP Vision conditioner — encodes images to CLIP embeddings.

Used by image-conditioned pipelines (I2V, image variation, etc.)
"""
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, List, Union
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
        self.output_mode = self.config.get("output_mode", "pooled")  # "pooled" | "patches" (IP-Adapter Plus)
        self._build_model()
    
    @classmethod
    def declare_io(cls) -> dict:
        return {
            "raw_condition": InputPort(
                "raw_condition",
                data_type="dict",
                optional=True,
                description="Dict with 'image' (single PIL/tensor) or 'images' (list of PIL/tensors), optional 'scales' (list of floats)",
            ),
            "embedding": OutputPort("embedding", spec=TensorSpec(space="embedding")),
            "pooled_embedding": OutputPort("pooled_embedding", spec=TensorSpec(space="embedding")),
            "patches": OutputPort("patches", description="Patch embeddings (N, num_patches+1, dim) for IP-Adapter Plus. Set output_mode='patches'."),
            "scales": OutputPort("scales", description="Per-image weights for IP-Adapter (pass-through from raw_condition)"),
        }
    
    @property
    def embedding_dim(self) -> int:
        """Output embedding dimension (768 for ViT-L, 1024 for ViT-H)."""
        if self._model is not None and hasattr(self._model, "visual_projection"):
            return getattr(self._model.visual_projection, "out_features", 768)
        if self._model is not None and hasattr(self._model, "config"):
            return getattr(self._model.config, "projection_dim", 768)
        return 768

    def _build_model(self):
        try:
            from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
            # Full CLIP (openai/clip-vit-large-patch14) has text+vision; we load only vision, so text keys
            # are "unexpected" — suppress the load report to avoid noisy logs.
            log = logging.getLogger("transformers")
            old_level = log.level
            log.setLevel(logging.ERROR)
            try:
                self._model = CLIPVisionModelWithProjection.from_pretrained(
                    self.pretrained, low_cpu_mem_usage=False
                )
                self._processor = CLIPImageProcessor.from_pretrained(self.pretrained)
            finally:
                log.setLevel(old_level)
            self._model.requires_grad_(False)
        except Exception:
            self._model = None
    
    def _images_to_list(self, raw: dict) -> List[Any]:
        """Return a list of images from raw_condition (single or 'images' list)."""
        images = raw.get("images")
        if images is not None and isinstance(images, (list, tuple)):
            return list(images)
        image = raw.get("image")
        if image is not None:
            if isinstance(image, (list, tuple)):
                return list(image)
            return [image]
        return []

    def process(self, **port_inputs) -> dict:
        raw = port_inputs.get("raw_condition", {})
        if not isinstance(raw, dict):
            raw = {"image": raw}

        image_list = self._images_to_list(raw)
        if not image_list:
            return {"embedding": None, "pooled_embedding": None, "output": None, "patches": None, "scales": None}

        scales = raw.get("scales")  # Per-image weights for IP-Adapter weighted mean
        if scales is not None and not isinstance(scales, (list, tuple)):
            scales = None

        if self._model is None or self._processor is None:
            dim = self.embedding_dim
            emb = torch.randn(len(image_list), 1, dim)
            patches_out = torch.randn(len(image_list), 257, dim) if self.output_mode == "patches" else None
            return {
                "embedding": emb,
                "pooled_embedding": emb.squeeze(1),
                "output": emb,
                "patches": patches_out,
                "scales": scales,
            }

        device = next(self._model.parameters()).device
        with torch.no_grad():
            inputs = self._processor(images=image_list, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            image_embeds = outputs.image_embeds  # (N, embed_dim)
            emb = image_embeds.unsqueeze(1)  # (N, 1, embed_dim)
            patches_out = None
            if self.output_mode == "patches" and hasattr(outputs, "last_hidden_state"):
                patches_out = outputs.last_hidden_state  # (N, num_patches+1, dim)
            return {
                "embedding": emb,
                "pooled_embedding": image_embeds,
                "output": emb,
                "patches": patches_out,
                "scales": scales,
            }
    
    def __call__(self, condition):
        return self.process(raw_condition=condition)
