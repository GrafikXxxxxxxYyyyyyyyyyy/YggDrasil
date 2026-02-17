# yggdrasil/core/graph/model_graph_builder.py
"""Build a ComputeGraph from a model/modular config (graph engine only, no slots).

Replaces slot-based model construction: the model is always a DAG of blocks
wired by ports. This module turns the same YAML structure (backbone, codec,
conditioner, guidance, position, adapters, diffusion_process) into a ComputeGraph.
"""
from __future__ import annotations

from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List

from yggdrasil.core.block.builder import BlockBuilder
from yggdrasil.core.graph.graph import ComputeGraph


# Default configs for optional model components (same semantics as legacy slots)
_DEFAULT_CODEC = {"type": "codec/identity"}
_DEFAULT_GUIDANCE = {"type": "guidance/cfg"}
_DEFAULT_POSITION = {"type": "position/rope_nd"}
_DEFAULT_DIFFUSION_PROCESS = {"type": "diffusion/process/rectified_flow"}


def build_model_graph(config: DictConfig | dict, name: str = "model") -> ComputeGraph:
    """Build a ComputeGraph for a diffusion model from config.

    Config keys (all optional except backbone):
        backbone: required, backbone block config
        codec: optional, latent codec (default: codec/identity)
        conditioner: optional, list or single conditioner configs
        guidance: optional, list or single guidance configs (default: guidance/cfg)
        position: optional, position embedder config
        adapters: optional, list of adapter configs
        diffusion_process: optional, diffusion process config

    Returns:
        ComputeGraph with nodes codec, position, conditioner_0..n, backbone,
        guidance_0..n, adapter_0..n, and edges matching the diffusion pipeline.
    """
    if isinstance(config, dict):
        config = OmegaConf.create(config)
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))

    graph = ComputeGraph(name)

    # ── Build blocks from config (no Slot; we build via BlockBuilder) ──
    backbone_cfg = config.get("backbone")
    if not backbone_cfg:
        raise ValueError("Model config must contain 'backbone'")
    backbone = BlockBuilder.build(backbone_cfg)
    graph.add_node("backbone", backbone)

    # Codec (optional)
    codec_cfg = config.get("codec", _DEFAULT_CODEC)
    if codec_cfg is not None:
        if isinstance(codec_cfg, dict) and (codec_cfg.get("type") or codec_cfg.get("block_type")):
            codec = BlockBuilder.build(codec_cfg)
            graph.add_node("codec", codec)

    # Position (optional)
    position_cfg = config.get("position", _DEFAULT_POSITION)
    if position_cfg is not None:
        if isinstance(position_cfg, dict) and (position_cfg.get("type") or position_cfg.get("block_type")):
            position = BlockBuilder.build(position_cfg)
            graph.add_node("position", position)

    # Conditioners (optional, list or single)
    conditioner_configs: List[Dict[str, Any]] = []
    raw_cond = config.get("conditioner")
    if raw_cond is not None:
        if isinstance(raw_cond, (list, tuple)):
            conditioner_configs = [c for c in raw_cond if isinstance(c, dict) and (c.get("type") or c.get("block_type"))]
        elif isinstance(raw_cond, dict) and (raw_cond.get("type") or raw_cond.get("block_type")):
            conditioner_configs = [raw_cond]
    for i, ccfg in enumerate(conditioner_configs):
        cond = BlockBuilder.build(ccfg)
        graph.add_node(f"conditioner_{i}", cond)

    # Guidance (optional, list or single; default one CFG)
    guidance_configs: List[Dict[str, Any]] = []
    raw_guid = config.get("guidance", _DEFAULT_GUIDANCE)
    if raw_guid is not None:
        if isinstance(raw_guid, (list, tuple)):
            guidance_configs = [g for g in raw_guid if isinstance(g, dict) and (g.get("type") or g.get("block_type"))]
        elif isinstance(raw_guid, dict) and (raw_guid.get("type") or raw_guid.get("block_type")):
            guidance_configs = [raw_guid]
    if not guidance_configs:
        guidance_configs = [OmegaConf.create(_DEFAULT_GUIDANCE)]
    for i, gcfg in enumerate(guidance_configs):
        g = BlockBuilder.build(gcfg)
        if hasattr(g, "_backbone_ref"):
            g._backbone_ref = backbone
        graph.add_node(f"guidance_{i}", g)

    # Adapters (optional)
    adapter_configs: List[Dict[str, Any]] = []
    raw_adapt = config.get("adapters")
    if raw_adapt is not None:
        if isinstance(raw_adapt, (list, tuple)):
            adapter_configs = [a for a in raw_adapt if isinstance(a, dict) and (a.get("type") or a.get("block_type"))]
        elif isinstance(raw_adapt, dict) and (raw_adapt.get("type") or raw_adapt.get("block_type")):
            adapter_configs = [raw_adapt]
    for i, acfg in enumerate(adapter_configs):
        adapter = BlockBuilder.build(acfg)
        graph.add_node(f"adapter_{i}", adapter)

    # Diffusion process is not a node in the forward graph; it's used by the solver.
    # We don't add it as a node here; the model graph is encode -> position -> condition -> backbone -> guidance.

    # ── Wire graph ──
    graph.expose_input("x", "backbone", "x")
    graph.expose_input("timestep", "backbone", "timestep")

    if "position" in graph.nodes:
        graph.expose_input("timestep_pos", "position", "timestep")
        graph.connect("position", "embedding", "backbone", "position_embedding")

    n_cond = len(conditioner_configs)
    for i in range(n_cond):
        graph.expose_input("condition", f"conditioner_{i}", "raw_condition")
        graph.connect(f"conditioner_{i}", "embedding", "backbone", "condition")
    # When multiple conditioners connect to backbone.condition, executor passes a list;
    # backbone should merge list of dicts into one (see AbstractBackbone.process).

    prev_node = "backbone"
    prev_port = "output"
    n_guid = len(guidance_configs)
    for i in range(n_guid):
        g_node = f"guidance_{i}"
        graph.connect(prev_node, prev_port, g_node, "model_output")
        graph.expose_input("x", g_node, "x")
        graph.expose_input("timestep", g_node, "t")
        prev_node = g_node
        prev_port = "guided_output"

    graph.expose_output("noise_pred", prev_node, prev_port)
    return graph
