"""Helpers for parameter handling in serving (Gradio, API).

No heavy dependencies (no torch) — safe to import for tests.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Tuple


def infer_input_visibility(template_name: str, modality: str) -> Tuple[bool, bool, bool]:
    """Infer visibility of control_image, ip_image, source_image from template. G3 dynamic widgets."""
    name = (template_name or "").lower()
    has_control = "controlnet" in name or "control" in name or "canny" in name or "depth" in name
    has_source = "img2img" in name or "inpainting" in name or "img2vid" in name or modality == "video"
    return (has_control, True, has_source)


def merge_extra_params_json(kwargs: Dict[str, Any], extra_json: str) -> Dict[str, Any]:
    """Merge extra params from JSON string into kwargs. G3: dynamic graph_inputs.

    Only adds keys not already in kwargs or where kwargs value is None.
    Does not modify kwargs in place; returns updated copy.
    """
    result = dict(kwargs)
    if not extra_json or not extra_json.strip() or extra_json.strip() == "{}":
        return result
    try:
        extra = json.loads(extra_json)
        if isinstance(extra, dict):
            for k, v in extra.items():
                if k not in result or result[k] is None:
                    result[k] = v
    except (json.JSONDecodeError, TypeError):
        pass
    return result
