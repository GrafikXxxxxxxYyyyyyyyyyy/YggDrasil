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
    "loop/": "denoise_loop",
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

# Conditioner: only type + pretrained needed; rest from defaults
CONDITIONER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "conditioner/clip_sdxl": {"force_zeros_for_empty_prompt": True},
}

# Loop (denoise): num_steps and guidance_scale come from graph.metadata or pipeline at run time; never from add_node
LOOP_PARAMS_FROM_METADATA = ("num_steps", "guidance_scale")

# Codec: fp16, scaling_factor, latent_channels, spatial_scale_factor pulled by base_model / metadata
CODEC_DEFAULT_BY_BASE: Dict[str, Dict[str, Any]] = {
    "sdxl": {"fp16": True, "scaling_factor": 0.13025, "latent_channels": 4, "spatial_scale_factor": 8},
    "sd15": {"fp16": True, "scaling_factor": 0.18215, "latent_channels": 4, "spatial_scale_factor": 8},
    "sd3": {"fp16": True, "latent_channels": 16, "spatial_scale_factor": 8},
    "flux": {"fp16": True, "latent_channels": 16, "spatial_scale_factor": 8},
}


def get_default_config_for_block_type(block_type: str, graph_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Default config for a block type. graph_metadata used for pretrained, num_steps, codec params, etc."""
    out: Dict[str, Any] = {}
    meta = graph_metadata or {}

    if block_type in ADAPTER_DEFAULT_CONFIG:
        out = dict(ADAPTER_DEFAULT_CONFIG[block_type])
    if block_type in CONDITIONER_DEFAULTS:
        out = {**out, **CONDITIONER_DEFAULTS[block_type]}

    if block_type.startswith("loop/"):
        out["num_steps"] = int(meta.get("default_num_steps", 50))
        out["guidance_scale"] = float(meta.get("default_guidance_scale", 7.5))

    if block_type.startswith("codec/"):
        base = (meta.get("base_model") or "").lower()
        for key in ("sdxl", "sd15", "sd3", "flux"):
            if key in base:
                out.update(CODEC_DEFAULT_BY_BASE.get(key, CODEC_DEFAULT_BY_BASE["sd15"]))
                break
        else:
            out.setdefault("fp16", True)
            out.setdefault("latent_channels", 4)
            out.setdefault("spatial_scale_factor", 8)

    by_base = DEFAULT_PRETRAINED_BY_BASE.get(block_type)
    if by_base and meta:
        base = (meta.get("base_model") or "").lower()
        if "sdxl" in base:
            p = by_base.get("sdxl") or by_base.get("sd15")
        elif base:
            p = by_base.get("sd15") or by_base.get("sdxl")
        else:
            p = None
        if p:
            out.setdefault("pretrained", p)
    return out


# Backbone -> denoise loop: when user adds a backbone, we wrap it in the right loop automatically
BACKBONE_TO_LOOP: Dict[str, Dict[str, Tuple[str, List[str]]]] = {
    "backbone/unet2d_condition": {
        "sdxl": ("loop/denoise_sdxl", ["pretrained", "num_steps", "guidance_scale", "fp16"]),
        "sd15": ("loop/denoise_sdxl", ["pretrained", "num_steps", "guidance_scale", "fp16"]),  # same loop shape, SD1.5 pretrained
    },
}


def resolve_loop_for_backbone(
    backbone_type: str,
    config: Dict[str, Any],
    graph_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """If this backbone should be wrapped in a denoise loop, return (loop_type, loop_config). Else None."""
    meta = graph_metadata or {}
    by_backbone = BACKBONE_TO_LOOP.get(backbone_type)
    if not by_backbone:
        return None
    base = (meta.get("base_model") or "").lower()
    if "sdxl" in base:
        key = "sdxl"
    elif "sd15" in base or "sd1.5" in base:
        key = "sd15"
    else:
        # Default to sdxl so backbone is always wrapped when type is backbone/unet2d_condition
        key = "sdxl"
    if key not in by_backbone:
        return None
    loop_type, keys = by_backbone[key]
    loop_cfg = {k: config.get(k) for k in keys if k in config}
    loop_cfg["type"] = loop_type
    loop_cfg.setdefault("pretrained", config.get("pretrained"))
    loop_cfg.setdefault("num_steps", meta.get("default_num_steps", 50))
    loop_cfg.setdefault("guidance_scale", meta.get("default_guidance_scale", 7.5))
    loop_cfg.setdefault("fp16", True)
    return (loop_type, loop_cfg)


def resolve_inner_target_for_adapter(graph_nodes: Any, block_type: str) -> Optional[str]:
    """If this adapter should go into an inner graph (denoise loop node), return that node name. Else None.
    Only ControlNet and T2I-Adapter go inside the loop; IP-Adapter stays at top level.
    The loop node may be named "denoise_loop" or e.g. "MyAwesomeBackbone" (when user added a backbone).
    """
    if not block_type.startswith("adapter/"):
        return None
    if block_type == "adapter/ip_adapter":
        return None
    if "denoise_loop" in graph_nodes:
        return "denoise_loop"
    for node_name, block in (graph_nodes.items() if hasattr(graph_nodes, "items") else []):
        bt = getattr(block, "block_type", "")
        if bt.startswith("loop/"):
            return node_name
    return None
