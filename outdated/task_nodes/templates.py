"""
Graph templates (e.g. text_to_image) for Graph.from_template.

Canon: WorldGenerator_2.0/TODO_02, Graph_Level. Templates return config dict
for Graph.from_config; use identity stubs when no block_type specified.
"""

from __future__ import annotations

from typing import Any, Dict

from yggdrasill.foundation.graph import Graph
from yggdrasill.task_nodes.stubs import register_task_node_stubs


def _text_to_image_config(
    *,
    graph_id: str = "text_to_image",
    tokenizer_type: str = "tokenizer/identity",
    conditioner_type: str = "conditioner/identity",
    backbone_type: str = "backbone/identity",
    solver_type: str = "solver/identity",
    codec_type: str = "codec/identity",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Config for a minimal text-to-image graph: tokenizer -> conditioner -> backbone <-> solver, codec.
    Exposed: input text (-> tokenizer), image out (codec decode_image).
    """
    return {
        "graph_id": graph_id,
        "nodes": [
            {"node_id": "tokenizer", "block_type": tokenizer_type, "config": {}},
            {"node_id": "conditioner", "block_type": conditioner_type, "config": {}},
            {"node_id": "backbone", "block_type": backbone_type, "config": {}},
            {"node_id": "solver", "block_type": solver_type, "config": {}},
            {"node_id": "codec", "block_type": codec_type, "config": {}},
        ],
        "edges": [
            ("tokenizer", "token_ids", "conditioner", "input"),
            ("conditioner", "embedding", "backbone", "condition"),
            ("backbone", "pred", "solver", "pred"),
            ("solver", "next_latent", "backbone", "latent"),
            ("solver", "next_timestep", "backbone", "timestep"),
            ("codec", "encode_latent", "solver", "latent"),
            ("solver", "next_latent", "codec", "decode_latent"),
        ],
        "exposed_inputs": [
            {"node_id": "tokenizer", "port_name": "text", "name": "text"},
            {"node_id": "solver", "port_name": "timestep", "name": "timestep"},
        ],
        "exposed_outputs": [
            {"node_id": "codec", "port_name": "decode_image", "name": "image"},
        ],
    }


def _normalize_config(c: Dict[str, Any]) -> Dict[str, Any]:
    """Convert edges from list of tuples to list of dicts if needed."""
    edges = c.get("edges", [])
    if not edges:
        return c
    out = dict(c)
    normalized = []
    for e in edges:
        if isinstance(e, (list, tuple)):
            normalized.append({
                "source_node": e[0],
                "source_port": e[1],
                "target_node": e[2],
                "target_port": e[3],
            })
        else:
            normalized.append(e)
    out["edges"] = normalized
    return out


def _register_templates() -> None:
    """Register template builders on Graph (idempotent)."""
    if "text_to_image" not in Graph._template_builders:

        def build_text_to_image(**kwargs: Any) -> Dict[str, Any]:
            return _normalize_config(_text_to_image_config(**kwargs))

        Graph._template_builders["text_to_image"] = build_text_to_image


# Register templates when module is imported so Graph.from_template knows them.
_register_templates()


def ensure_task_node_stubs(registry: Any = None) -> None:
    """Ensure identity stubs are registered (e.g. before from_template with default registry)."""
    register_task_node_stubs(registry)
