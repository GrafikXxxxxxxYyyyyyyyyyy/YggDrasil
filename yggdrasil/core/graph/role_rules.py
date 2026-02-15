# yggdrasil/core/graph/role_rules.py
"""Правила автоопределения роли и подключения для add_node(type="...").

Реестр: block_type -> role; правила подключения для каждой роли.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

# Роли узлов уровня 1 (ТЗ)
ROLES = (
    "backbone",
    "codec",
    "conditioner",
    "guidance",
    "solver",
    "adapter",
    "inner_module",
    "outer_module",
    "processor",
)

# Маппинг block_type -> role (по префиксу)
TYPE_TO_ROLE: Dict[str, str] = {
    "backbone/": "backbone",
    "codec/": "codec",
    "conditioner/": "conditioner",
    "guidance/": "guidance",
    "diffusion/solver/": "solver",
    "solver/": "solver",
    "adapter/": "adapter",
    "inner_module/": "inner_module",
    "outer_module/": "outer_module",
    "processor/": "processor",
    "schedule/": "solver",  # schedule merged into solver
}


def get_role_for_block_type(block_type: str) -> str:
    """Определить роль блока по block_type."""
    for prefix, role in sorted(TYPE_TO_ROLE.items(), key=lambda x: -len(x[0])):
        if block_type.startswith(prefix):
            return role
    return "unknown"


def get_connection_rules(role: str) -> Optional[Dict[str, Any]]:
    """Правила авто-подключения для роли.

    Returns:
        Dict с ключами: target_node, target_port, output_port, input_port,
        graph_input (опционально). None если авто-подключение не определено.
    """
    rules = {
        "inner_module": {
            "target_node": "backbone",
            "target_port": "adapter_features",
            "output_port": "adapter_features",
            "input_port": "control_image",
            "graph_input": "control_image",
        },
        "adapter": {
            # ControlNet, T2I-Adapter, etc. — feed into backbone.adapter_features (TZ §4.3)
            "target_node": "backbone",
            "target_port": "adapter_features",
            "output_port": "output",
            "input_port": "control_image",
            "graph_input": "control_image",
        },
        "conditioner": {
            "target_node": None,  # цепочка condition — зависит от шаблона
            "target_port": None,
            "output_port": "embedding",
            "input_port": "text",
            "graph_input": None,
        },
        "processor": {
            "target_node": None,
            "target_port": None,
            "output_port": "output",
            "input_port": "input",
            "graph_input": "input",
        },
        "outer_module": {
            "target_node": None,
            "target_port": None,
            "output_port": "output",
            "input_port": "input",
            "graph_input": "input",
        },
    }
    return rules.get(role)


# Default config per adapter type (optional params; user can override)
ADAPTER_DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "adapter/controlnet": {
        "conditioning_scale": 1.0,
        "fp16": True,
    },
    "adapter/ip_adapter": {
        "scale": 0.6,
        "image_encoder_pretrained": "openai/clip-vit-large-patch14",
    },
    "adapter/t2i_adapter": {
        "scale": 1.0,
        "fp16": True,
    },
}

# Default pretrained by base_model when not provided
DEFAULT_PRETRAINED_BY_BASE: Dict[str, Dict[str, str]] = {
    "adapter/controlnet": {
        "sdxl": "diffusers/controlnet-canny-sdxl-1.0",
        "sd15": "lllyasviel/control_v11p_sd15_canny",
    },
    "adapter/ip_adapter": {
        "sdxl": "h94/IP-Adapter",
        "sd15": "h94/IP-Adapter",
    },
}


def get_default_config_for_block_type(block_type: str, graph_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Default config for a block type. graph_metadata used e.g. for pretrained by base_model."""
    out = dict(ADAPTER_DEFAULT_CONFIG.get(block_type, {}))
    by_base = DEFAULT_PRETRAINED_BY_BASE.get(block_type)
    if by_base and graph_metadata:
        base = (graph_metadata.get("base_model") or "").lower()
        if "sdxl" in base:
            p = by_base.get("sdxl") or by_base.get("sd15")
        elif base:
            p = by_base.get("sd15") or by_base.get("sdxl")
        else:
            p = None
        if p:
            out.setdefault("pretrained", p)
    return out


def resolve_inner_target_for_adapter(graph_nodes: Any, block_type: str) -> Optional[str]:
    """If this adapter should go into an inner graph (e.g. denoise_loop), return its name. Else None.
    Only ControlNet and T2I-Adapter go inside the loop; IP-Adapter stays at top level (encoder + adapter).
    """
    if not block_type.startswith("adapter/"):
        return None
    if block_type == "adapter/ip_adapter":
        return None
    if "backbone" in graph_nodes:
        return None
    if "denoise_loop" in graph_nodes:
        return "denoise_loop"
    return None
