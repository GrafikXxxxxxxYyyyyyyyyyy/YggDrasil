# yggdrasil/serving/contract_bridge.py
"""Связка унифицированного контракта с UI/API.

Позволяет по графу получить описание входов/выходов для динамического Gradio и API.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    from yggdrasil.core.unified import (
        PipelineContract,
        infer_contract,
        Modality,
        DiffusionStep,
    )
except ImportError:
    PipelineContract = None
    infer_contract = None
    Modality = None
    DiffusionStep = None


def get_pipeline_contract(graph: Any) -> Optional["PipelineContract"]:
    """По графу вернуть унифицированный контракт или None при ошибке."""
    if infer_contract is None or graph is None:
        return None
    try:
        return infer_contract(graph)
    except Exception:
        return None


def contract_to_ui_hints(contract: "PipelineContract") -> Dict[str, Any]:
    """Из контракта сформировать подсказки для UI: какие поля показывать и тип.

    Returns:
        {
            "inputs": [{"name": "prompt", "type": "text", "required": True}, ...],
            "outputs": [{"name": "images", "type": "image"}, ...],
            "has_control": bool,
            "has_post_process": bool,
            "modality": "image" | "video" | ...
        }
    """
    if contract is None:
        return {"inputs": [], "outputs": [], "has_control": False, "has_post_process": False, "modality": "any"}
    modality = getattr(contract.modality, "value", str(contract.modality)) if contract.modality else "any"
    input_hints = []
    for name in contract.get_input_names():
        t = "text" if "prompt" in name or "condition" in name else "any"
        if "image" in name or "control_image" in name or "source_image" in name:
            t = "image"
        if "audio" in name:
            t = "audio"
        if "video" in name:
            t = "video"
        input_hints.append({"name": name, "type": t, "required": name in ("prompt", "condition")})
    output_hints = []
    for name in contract.get_output_names():
        t = "image" if "image" in name or name == "images" else "any"
        if "audio" in name:
            t = "audio"
        if "video" in name:
            t = "video"
        output_hints.append({"name": name, "type": t})
    return {
        "inputs": input_hints,
        "outputs": output_hints,
        "has_control": contract.has_control(),
        "has_post_process": contract.has_post_process(),
        "modality": modality,
    }
