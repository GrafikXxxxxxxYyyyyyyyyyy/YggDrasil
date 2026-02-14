# yggdrasil/core/graph/templates/control_pipelines.py
"""Graph templates for ControlNet, T2I-Adapter, IP-Adapter pipelines.

Each template builds a full pipeline with denoising loop.
"""
from __future__ import annotations
from yggdrasil.core.graph.graph import ComputeGraph
from yggdrasil.core.graph.templates import register_template


def _build_controlnet_step(backbone, guidance, solver, controlnet):
    """Inner step graph with ControlNet: controlnet -> backbone -> guidance -> solver."""
    step = ComputeGraph("controlnet_step")
    step.add_node("controlnet", controlnet)
    step.add_node("backbone", backbone)
    step.add_node("guidance", guidance)
    step.add_node("solver", solver)
    
    # ControlNet features feed into backbone
    step.connect("controlnet", "output", "backbone", "adapter_features")
    step.connect("backbone", "output", "guidance", "model_output")
    step.connect("guidance", "guided_output", "solver", "model_output")
    
    # Fan-out: latents
    step.expose_input("latents", "backbone", "x")
    step.expose_input("latents", "controlnet", "input")
    step.expose_input("latents", "solver", "current_latents")
    step.expose_input("latents", "guidance", "x")
    
    # Fan-out: timestep
    step.expose_input("timestep", "backbone", "timestep")
    step.expose_input("timestep", "solver", "timestep")
    step.expose_input("timestep", "guidance", "t")
    
    # Fan-out: condition
    step.expose_input("condition", "backbone", "condition")
    step.expose_input("condition", "controlnet", "context")
    step.expose_input("condition", "guidance", "condition")
    
    step.expose_input("next_timestep", "solver", "next_timestep")
    step.expose_input("control_image", "controlnet", "input")
    
    step.expose_output("next_latents", "solver", "next_latents")
    step.expose_output("latents", "solver", "next_latents")
    
    return step


@register_template("controlnet_txt2img")
def controlnet_txt2img(**kwargs) -> ComputeGraph:
    """ControlNet text-to-image pipeline with denoising loop."""
    from yggdrasil.core.block.builder import BlockBuilder
    from yggdrasil.core.graph.subgraph import LoopSubGraph
    
    pretrained = kwargs.get("pretrained", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    cn_pretrained = kwargs.get("controlnet_pretrained", "lllyasviel/control_v11p_sd15_canny")
    
    backbone = BlockBuilder.build({"type": "backbone/unet2d_condition", "pretrained": pretrained, "fp16": True})
    guidance = BlockBuilder.build({"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 7.5)})
    solver = BlockBuilder.build({"type": "diffusion/solver/ddim"})
    controlnet = BlockBuilder.build({"type": "adapter/controlnet", "pretrained": cn_pretrained, "fp16": True})
    conditioner = BlockBuilder.build({"type": "conditioner/clip_text", "pretrained": pretrained, "tokenizer_subfolder": "tokenizer", "text_encoder_subfolder": "text_encoder", "max_length": 77})
    codec = BlockBuilder.build({"type": "codec/autoencoder_kl", "pretrained": pretrained, "fp16": True, "scaling_factor": 0.18215, "latent_channels": 4})
    
    guidance._backbone_ref = backbone
    
    step = _build_controlnet_step(backbone, guidance, solver, controlnet)
    loop = LoopSubGraph.create(inner_graph=step, num_iterations=kwargs.get("num_steps", 50))
    
    graph = ComputeGraph("controlnet_txt2img")
    graph.add_node("conditioner_0", conditioner)
    graph.add_node("denoise_loop", loop)
    graph.add_node("codec", codec)
    
    graph.expose_input("prompt", "conditioner_0", "raw_condition")
    graph.connect("conditioner_0", "embedding", "denoise_loop", "condition")
    graph.expose_input("latents", "denoise_loop", "initial_latents")
    graph.expose_input("control_image", "denoise_loop", "control_image")
    graph.connect("denoise_loop", "latents", "codec", "latent")
    
    graph.expose_output("decoded", "codec", "decoded")
    graph.expose_output("latents", "denoise_loop", "latents")
    
    return graph


@register_template("t2i_adapter_txt2img")
def t2i_adapter_txt2img(**kwargs) -> ComputeGraph:
    """T2I-Adapter pipeline (same structure as ControlNet)."""
    graph = controlnet_txt2img(**kwargs)
    graph.name = "t2i_adapter_txt2img"
    return graph


@register_template("ip_adapter_txt2img")
def ip_adapter_txt2img(**kwargs) -> ComputeGraph:
    """IP-Adapter pipeline."""
    graph = controlnet_txt2img(**kwargs)
    graph.name = "ip_adapter_txt2img"
    graph.expose_input("image_prompt", "denoise_loop", "control_image")
    return graph


@register_template("controlnet_multi")
def controlnet_multi(**kwargs) -> ComputeGraph:
    """Multi-ControlNet pipeline."""
    graph = controlnet_txt2img(**kwargs)
    graph.name = "controlnet_multi"
    return graph
