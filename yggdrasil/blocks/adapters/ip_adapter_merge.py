"""IP-Adapter Merge — combines multiple image_prompt_embeds with per-adapter scales.

Lego block: multiple adapters -> weighted sum -> single image_prompt_embeds.
"""
from __future__ import annotations

import logging
import torch
from typing import Any, Dict, List
from omegaconf import DictConfig

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import InputPort, OutputPort
from yggdrasil.core.block.base import AbstractBaseBlock

logger = logging.getLogger(__name__)


@register_block("adapter/ip_adapter_merge")
class IPAdapterMerge(AbstractBaseBlock):
    """Merge multiple IP-Adapter outputs with per-adapter scales.

    Inputs: embeds (list of tensors) or embeds_0, embeds_1, ...
    Output: weighted sum. Scales from scales (list) or default 1.0.
    """

    block_type = "adapter/ip_adapter_merge"

    def __init__(self, config: DictConfig = None):
        config = config or {"type": "adapter/ip_adapter_merge"}
        super().__init__(config)
        self.default_scale = config.get("default_scale", 1.0)

    @classmethod
    def declare_io(cls):
        return {
            "embeds_0": InputPort("embeds_0", optional=True, description="IP-Adapter 0 output"),
            "embeds_1": InputPort("embeds_1", optional=True, description="IP-Adapter 1 output"),
            "embeds_2": InputPort("embeds_2", optional=True, description="IP-Adapter 2 output (FaceID, etc.)"),
            "embeds": InputPort("embeds", optional=True, description="List of all adapter outputs"),
            "scales": InputPort("scales", optional=True, description="Per-adapter scales [s0, s1, ...]"),
            "image_prompt_embeds": OutputPort("image_prompt_embeds", description="Merged tokens (concat along seq)"),
            "scales_out": OutputPort("scales_out", description="Per-adapter scales for UNet processor"),
        }

    def process(self, **kw) -> Dict[str, Any]:
        embeds_list = kw.get("embeds")
        if embeds_list is None:
            embeds_list = [kw.get("embeds_0"), kw.get("embeds_1"), kw.get("embeds_2")]
            embeds_list = [e for e in embeds_list if e is not None]
        elif not isinstance(embeds_list, (list, tuple)):
            embeds_list = [embeds_list]
        valid = [e for e in embeds_list if e is not None]
        if not valid:
            return {"image_prompt_embeds": None, "scales_out": None}
        scales = kw.get("scales")
        if scales is None:
            scales = [self.default_scale] * len(valid)
        elif not isinstance(scales, (list, tuple)):
            scales = [float(scales)] * len(valid)
        scales = [float(s) for s in scales[: len(valid)]]
        while len(scales) < len(valid):
            scales.append(self.default_scale)
        if len(valid) == 1:
            return {"image_prompt_embeds": valid[0], "scales_out": scales}
        out = torch.cat(valid, dim=1)
        return {"image_prompt_embeds": out, "scales_out": scales}
