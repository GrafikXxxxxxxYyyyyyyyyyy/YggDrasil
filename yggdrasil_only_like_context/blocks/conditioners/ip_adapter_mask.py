"""IP-Adapter mask preprocessor — binarize and resize masks for region-specific IP conditioning.

Ref: Diffusers IPAdapterMaskProcessor. Masks define which image regions get which IP-Adapter image.
Used with cross_attention_kwargs={"ip_adapter_masks": masks} when the UNet processor supports it.
"""
from __future__ import annotations

import torch
from typing import Any, Dict, List, Union
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.block.port import InputPort, OutputPort
from ...core.block.base import AbstractBaseBlock


@register_block("conditioner/ip_adapter_mask")
class IPAdapterMaskConditioner(AbstractBaseBlock):
    """Preprocess binary masks for IP-Adapter region-specific conditioning.

    Masks: list of PIL/tensor (H,W) or (1,H,W). Values >0.5 → 1, else 0.
    Output: (num_masks, 1, h, w) resized to height/width for cross_attention.
    """

    block_type = "conditioner/ip_adapter_mask"

    @classmethod
    def declare_io(cls) -> dict:
        return {
            "masks": InputPort("masks", data_type="any", description="List of mask images (PIL or tensor)"),
            "height": InputPort("height", data_type="any", optional=True, description="Target height (default 64 for latent)"),
            "width": InputPort("width", data_type="any", optional=True, description="Target width"),
            "output": OutputPort("output", description="Preprocessed masks (N, 1, h, w) float32"),
        }

    def process(self, **port_inputs: Any) -> Dict[str, Any]:
        masks = port_inputs.get("masks")
        height = port_inputs.get("height", 64)
        width = port_inputs.get("width", 64)
        if masks is None or (isinstance(masks, (list, tuple)) and not masks):
            return {"output": None}
        if not isinstance(masks, (list, tuple)):
            masks = [masks]
        h = int(height) if height is not None else 64
        w = int(width) if width is not None else 64
        out = []
        for m in masks:
            if hasattr(m, "size"):
                from PIL import Image
                import numpy as np
                arr = np.array(m.convert("L")).astype(np.float32) / 255.0
                t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
            elif isinstance(m, torch.Tensor):
                t = m.float()
                if t.dim() == 2:
                    t = t.unsqueeze(0).unsqueeze(0)
                elif t.dim() == 3:
                    t = t[:1].unsqueeze(0)
            else:
                continue
            t = (t > 0.5).float()
            t = torch.nn.functional.interpolate(t, size=(h, w), mode="nearest")
            out.append(t)
        if not out:
            return {"output": None}
        stacked = torch.cat(out, dim=0)
        return {"output": stacked}
