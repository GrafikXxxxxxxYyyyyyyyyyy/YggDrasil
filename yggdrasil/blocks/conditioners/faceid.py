"""FaceID encoder — InsightFace face embeddings for IP-Adapter FaceID.

Lego block: raw face image -> face embedding (512-dim).
Requires: pip install insightface
"""
from __future__ import annotations

import logging
import torch
from typing import Any, Dict, List
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.block.port import InputPort, OutputPort, TensorSpec
from ...core.model.conditioner import AbstractConditioner

logger = logging.getLogger(__name__)


@register_block("conditioner/faceid")
class FaceIDConditioner(AbstractConditioner):
    """InsightFace face encoder for IP-Adapter FaceID."""

    block_type = "conditioner/faceid"

    def __init__(self, config=None):
        config = config or {"type": "conditioner/faceid"}
        super().__init__(config)
        self._app = None
        self.providers = config.get("providers", ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.det_size = config.get("det_size", (640, 640))
        self._build()

    def _build(self):
        try:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(name="buffalo_l", providers=list(self.providers))
            self._app.prepare(ctx_id=0, det_size=self.det_size)
        except ImportError as e:
            logger.warning("FaceID requires insightface: pip install insightface. %s", e)
            self._app = None

    @classmethod
    def declare_io(cls):
        return {
            "raw_condition": InputPort("raw_condition", data_type="dict", optional=True,
                description="Dict with 'image' (PIL/tensor) or 'images' (list). Face crops."),
            "embedding": OutputPort("embedding", spec=TensorSpec(space="embedding"),
                description="Face embeddings (N, 1, 512) for IP-Adapter FaceID"),
        }

    @property
    def embedding_dim(self):
        return 512

    def _images_to_list(self, raw):
        imgs = raw.get("images")
        if imgs and isinstance(imgs, (list, tuple)):
            return list(imgs)
        img = raw.get("image")
        if img is not None:
            return [img] if not isinstance(img, (list, tuple)) else list(img)
        return []

    def process(self, **port_inputs):
        raw = port_inputs.get("raw_condition", {})
        if not isinstance(raw, dict):
            raw = {"image": raw}
        images = self._images_to_list(raw)
        if not images:
            return {"embedding": None}
        if self._app is None:
            return {"embedding": torch.randn(len(images), 1, 512)}
        import numpy as np
        emb_list = []
        for img in images:
            if hasattr(img, "convert"):
                arr = np.array(img.convert("RGB"))
            elif isinstance(img, torch.Tensor):
                arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            else:
                emb_list.append(torch.zeros(512))
                continue
            faces = self._app.get(arr)
            if not faces:
                emb_list.append(torch.zeros(512))
                continue
            emb = torch.from_numpy(faces[0].normed_embedding).float()
            emb_list.append(emb)
        stacked = torch.stack(emb_list).unsqueeze(1)
        return {"embedding": stacked}
