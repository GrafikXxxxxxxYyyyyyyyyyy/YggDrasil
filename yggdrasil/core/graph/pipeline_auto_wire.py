# yggdrasil/core/graph/pipeline_auto_wire.py
"""Auto-wire pipeline when adding nodes by role: conditioner -> denoise_loop -> codec.

Developer can call add_node() 5 times (conditioner, denoise_loop, codec, + adapters)
and get a working graph without manual connect/expose. For custom pipelines,
manual connect/expose_input/expose_output remain available.
"""
from __future__ import annotations

import logging
from typing import Any, List, Tuple

from .role_rules import get_role_for_block_type

logger = logging.getLogger(__name__)


def _get_block_type(block: Any) -> str:
    return getattr(block, "block_type", "")


def _edge_exists(graph: Any, src: str, src_port: str, dst: str, dst_port: str) -> bool:
    for e in graph.edges:
        if e.src_node == src and e.src_port == src_port and e.dst_node == dst and e.dst_port == dst_port:
            return True
    return False


def _input_exposed(graph: Any, name: str) -> bool:
    return name in graph.graph_inputs


def _output_exposed(graph: Any, name: str) -> bool:
    return name in graph.graph_outputs


def _nodes_by_role(graph: Any) -> dict[str, List[Tuple[str, str]]]:
    """Return role -> [(node_name, block_type), ...] for pipeline nodes (no adapters)."""
    role_to_nodes: dict[str, List[Tuple[str, str]]] = {}
    for name, block in graph.nodes.items():
        bt = _get_block_type(block)
        if not bt:
            continue
        role = get_role_for_block_type(bt)
        if role in ("conditioner", "denoise_loop", "codec"):
            role_to_nodes.setdefault(role, []).append((name, bt))
    return role_to_nodes


def apply_pipeline_auto_wire(graph: Any) -> None:
    """Connect conditioner -> denoise_loop -> codec and expose inputs/outputs.

    Idempotent: only adds edges and expose that do not already exist.
    Single conditioner with condition+uncond (e.g. clip_sdxl): connects both to denoise_loop.
    Two conditioners (CFG): first -> condition, second -> uncond.
    """
    by_role = _nodes_by_role(graph)
    conditioners = by_role.get("conditioner", [])
    denoise_loops = by_role.get("denoise_loop", [])
    codecs = by_role.get("codec", [])

    if not denoise_loops:
        return
    denoise_name = denoise_loops[0][0]
    denoise_bt = denoise_loops[0][1]

    # Conditioner(s) -> denoise_loop
    if conditioners:
        if len(conditioners) == 1:
            cond_name, cond_bt = conditioners[0]
            if "clip_sdxl" in cond_bt or "clip_sdxl" in _get_block_type(graph.nodes[cond_name]):
                if not _edge_exists(graph, cond_name, "condition", denoise_name, "condition"):
                    graph.connect(cond_name, "condition", denoise_name, "condition")
                if not _edge_exists(graph, cond_name, "uncond", denoise_name, "uncond"):
                    graph.connect(cond_name, "uncond", denoise_name, "uncond")
            else:
                if not _edge_exists(graph, cond_name, "embedding", denoise_name, "condition"):
                    graph.connect(cond_name, "embedding", denoise_name, "condition")
            _expose_conditioner_inputs(graph, conditioners)
        else:
            c0_name = conditioners[0][0]
            c1_name = conditioners[1][0]
            if not _edge_exists(graph, c0_name, "embedding", denoise_name, "condition"):
                graph.connect(c0_name, "embedding", denoise_name, "condition")
            if not _edge_exists(graph, c1_name, "embedding", denoise_name, "uncond"):
                graph.connect(c1_name, "embedding", denoise_name, "uncond")
            if not _input_exposed(graph, "prompt"):
                graph.expose_input("prompt", c0_name, "raw_condition" if _port_exists(graph, c0_name, "raw_condition") else "prompt")
            if not _input_exposed(graph, "negative_prompt"):
                graph.expose_input("negative_prompt", c1_name, "raw_condition" if _port_exists(graph, c1_name, "raw_condition") else "prompt")
            _expose_conditioner_inputs(graph, conditioners, skip=("prompt", "negative_prompt"))

    # denoise_loop inputs
    if not _input_exposed(graph, "latents"):
        graph.expose_input("latents", denoise_name, "initial_latents")
    if not _input_exposed(graph, "timesteps"):
        graph.expose_input("timesteps", denoise_name, "timesteps")

    # denoise_loop -> codec
    if codecs:
        codec_name = codecs[0][0]
        if not _edge_exists(graph, denoise_name, "latents", codec_name, "latent"):
            graph.connect(denoise_name, "latents", codec_name, "latent")
        if not _output_exposed(graph, "decoded"):
            graph.expose_output("decoded", codec_name, "decoded")
    if not _output_exposed(graph, "latents"):
        graph.expose_output("latents", denoise_name, "latents")


def _port_exists(graph: Any, node: str, port: str) -> bool:
    block = graph.nodes.get(node)
    if not block:
        return False
    # Use class-level declare_io so we get the declared ports (works for @classmethod)
    declare_io = getattr(block.__class__, "declare_io", None)
    if declare_io is None or not callable(declare_io):
        return False
    try:
        io = declare_io()
    except Exception:
        return False
    return isinstance(io, dict) and port in io


def _expose_conditioner_inputs(
    graph: Any,
    conditioners: List[Tuple[str, str]],
    skip: Tuple[str, ...] = (),
) -> None:
    if not conditioners:
        return
    cond_name, cond_bt = conditioners[0]
    block = graph.nodes.get(cond_name)
    bt = _get_block_type(block) or cond_bt or ""
    # Prefer "prompt" for clip_sdxl; fallback to "raw_condition" for other conditioners
    prompt_port = "prompt" if "clip_sdxl" in bt else "raw_condition"
    for inp, port in (
        ("prompt", prompt_port),
        ("negative_prompt", "negative_prompt"),
        ("height", "height"),
        ("width", "width"),
    ):
        if inp in skip or _input_exposed(graph, inp):
            continue
        if _port_exists(graph, cond_name, port):
            graph.expose_input(inp, cond_name, port)
    # Ensure "prompt" is always exposed for single conditioner (try fallback port if main not present)
    if "prompt" not in skip and not _input_exposed(graph, "prompt"):
        for try_port in (prompt_port, "prompt", "raw_condition"):
            if try_port != prompt_port and _port_exists(graph, cond_name, try_port):
                graph.expose_input("prompt", cond_name, try_port)
                break
