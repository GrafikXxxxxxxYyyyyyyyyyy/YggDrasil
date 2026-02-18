# yggdrasil/core/graph/templates/specialized_pipelines.py
"""Graph templates for specialized pipelines: 3D, molecular, timeseries.

All pipelines use LoopSubGraph for proper denoising.
"""
from __future__ import annotations
from yggdrasil.core.graph.graph import ComputeGraph
from yggdrasil.core.graph.templates import register_template
from yggdrasil.core.graph.templates.image_pipelines import _build_denoise_step


@register_template("shap_e_txt2_3d")
def shap_e_txt2_3d(**kwargs) -> ComputeGraph:
    """Shap-E text-to-3D pipeline."""
    from yggdrasil.core.block.builder import BlockBuilder
    from yggdrasil.core.graph.subgraph import LoopSubGraph
    
    backbone = BlockBuilder.build({"type": "backbone/dit", "hidden_dim": 1024, "num_layers": 16, "num_heads": 16, "in_channels": 1024, "patch_size": 1})
    guidance = BlockBuilder.build({"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 15.0)})
    solver = BlockBuilder.build({"type": "diffusion/solver/ddim"})
    conditioner = BlockBuilder.build({"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14"})
    
    guidance._backbone_ref = backbone
    
    step = _build_denoise_step(backbone, guidance, solver)
    loop = LoopSubGraph.create(inner_graph=step, num_iterations=kwargs.get("num_steps", 64))
    
    graph = ComputeGraph("shap_e_txt2_3d")
    graph.add_node("conditioner_0", conditioner)
    graph.add_node("denoise_loop", loop)
    
    graph.expose_input("prompt", "conditioner_0", "raw_condition")
    graph.connect("conditioner_0", "embedding", "denoise_loop", "condition")
    graph.expose_input("latents", "denoise_loop", "initial_latents")
    
    graph.expose_output("latents", "denoise_loop", "latents")
    
    return graph


@register_template("molecular_edm")
def molecular_edm(**kwargs) -> ComputeGraph:
    """Molecular EDM (equivariant diffusion for molecule generation)."""
    from yggdrasil.core.block.builder import BlockBuilder
    from yggdrasil.core.graph.subgraph import LoopSubGraph
    
    backbone = BlockBuilder.build({"type": "backbone/equivariant_gnn", "hidden_dim": 256, "num_layers": 6, "num_heads": 8})
    solver = BlockBuilder.build({"type": "diffusion/solver/ddim"})
    
    # No guidance for molecular EDM (unconditional)
    step = ComputeGraph("edm_step")
    step.add_node("backbone", backbone)
    step.add_node("solver", solver)
    step.connect("backbone", "output", "solver", "model_output")
    step.expose_input("latents", "backbone", "x")
    step.expose_input("latents", "solver", "current_latents")
    step.expose_input("timestep", "backbone", "timestep")
    step.expose_input("timestep", "solver", "timestep")
    step.expose_input("next_timestep", "solver", "next_timestep")
    step.expose_output("next_latents", "solver", "next_latents")
    step.expose_output("latents", "solver", "next_latents")
    
    loop = LoopSubGraph.create(inner_graph=step, num_iterations=kwargs.get("num_steps", 1000))
    
    graph = ComputeGraph("molecular_edm")
    graph.add_node("denoise_loop", loop)
    
    graph.expose_input("latents", "denoise_loop", "initial_latents")
    graph.expose_output("molecules", "denoise_loop", "latents")
    
    return graph


@register_template("timeseries_forecast")
def timeseries_forecast(**kwargs) -> ComputeGraph:
    """Time series forecasting via diffusion."""
    from yggdrasil.core.block.builder import BlockBuilder
    from yggdrasil.core.graph.subgraph import LoopSubGraph
    
    backbone = BlockBuilder.build({"type": "backbone/transformer_1d", "hidden_dim": 256, "num_layers": 4, "num_heads": 4, "in_channels": 1})
    guidance = BlockBuilder.build({"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 1.0)})
    solver = BlockBuilder.build({"type": "diffusion/solver/ddim"})
    
    guidance._backbone_ref = backbone
    
    step = _build_denoise_step(backbone, guidance, solver)
    loop = LoopSubGraph.create(inner_graph=step, num_iterations=kwargs.get("num_steps", 100))
    
    graph = ComputeGraph("timeseries_forecast")
    graph.add_node("denoise_loop", loop)
    
    graph.expose_input("latents", "denoise_loop", "initial_latents")
    graph.expose_input("condition", "denoise_loop", "condition")
    graph.expose_output("forecast", "denoise_loop", "latents")
    
    return graph
