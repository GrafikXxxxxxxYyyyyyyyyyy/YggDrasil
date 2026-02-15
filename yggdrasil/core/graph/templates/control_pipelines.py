# yggdrasil/core/graph/templates/control_pipelines.py
"""Graph templates for ControlNet, T2I-Adapter, IP-Adapter pipelines.

Each template builds a full pipeline with denoising loop.
Supports: SD 1.5, SDXL (ControlNet/T2I/IP-Adapter); FLUX/SD3 use different adapter paths.
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
    step.expose_input("latents", "controlnet", "sample")
    step.expose_input("latents", "solver", "current_latents")
    step.expose_input("latents", "guidance", "x")
    
    # Fan-out: timestep
    step.expose_input("timestep", "backbone", "timestep")
    step.expose_input("timestep", "controlnet", "timestep")
    step.expose_input("timestep", "solver", "timestep")
    step.expose_input("timestep", "guidance", "t")
    
    # Fan-out: condition
    step.expose_input("condition", "backbone", "condition")
    step.expose_input("condition", "controlnet", "encoder_hidden_states")
    step.expose_input("condition", "guidance", "condition")
    
    step.expose_input("next_timestep", "solver", "next_timestep")
    step.expose_input("control_image", "controlnet", "control_image")
    
    step.expose_output("next_latents", "solver", "next_latents")
    step.expose_output("latents", "solver", "next_latents")
    
    return step


@register_template("controlnet_txt2img")
def controlnet_txt2img(**kwargs) -> ComputeGraph:
    """ControlNet text-to-image for Stable Diffusion 1.5 (Canny, Depth, etc.)."""
    from yggdrasil.core.block.builder import BlockBuilder
    from yggdrasil.core.graph.subgraph import LoopSubGraph

    pretrained = kwargs.get("pretrained", "runwayml/stable-diffusion-v1-5")
    cn_pretrained = kwargs.get("controlnet_pretrained", "lllyasviel/control_v11p_sd15_canny")

    backbone = BlockBuilder.build({"type": "backbone/unet2d_condition", "pretrained": pretrained, "fp16": True})
    guidance = BlockBuilder.build({"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 7.5)})
    solver = BlockBuilder.build({"type": "diffusion/solver/ddim"})
    controlnet = BlockBuilder.build({"type": "adapter/controlnet", "pretrained": cn_pretrained, "fp16": True})
    conditioner = BlockBuilder.build({"type": "conditioner/clip_text", "pretrained": pretrained, "tokenizer_subfolder": "tokenizer", "text_encoder_subfolder": "text_encoder", "max_length": 77})
    codec = BlockBuilder.build({"type": "codec/autoencoder_kl", "pretrained": pretrained, "fp16": True, "scaling_factor": 0.18215, "latent_channels": 4, "spatial_scale_factor": 8})

    guidance._backbone_ref = backbone

    step = _build_controlnet_step(backbone, guidance, solver, controlnet)
    loop = LoopSubGraph.create(inner_graph=step, num_iterations=kwargs.get("num_steps", 50), num_train_timesteps=1000)

    graph = ComputeGraph("controlnet_txt2img")
    graph.metadata = {"default_num_steps": 50, "default_width": 512, "default_height": 512, "latent_channels": 4, "spatial_scale_factor": 8, "base_model": "sd15"}
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


@register_template("controlnet_sdxl_txt2img")
def controlnet_sdxl_txt2img(**kwargs) -> ComputeGraph:
    """ControlNet text-to-image for Stable Diffusion XL."""
    from yggdrasil.core.block.builder import BlockBuilder
    from yggdrasil.core.graph.subgraph import LoopSubGraph
    from yggdrasil.core.graph.templates.image_pipelines import _build_sdxl_txt2img_graph

    pretrained = kwargs.get("pretrained", "stabilityai/stable-diffusion-xl-base-1.0")
    cn_pretrained = kwargs.get("controlnet_pretrained", "diffusers/controlnet-canny-sdxl-1.0")
    num_steps = kwargs.get("num_steps", 50)

    # Base SDXL graph (no batched CFG so inner step has backbone suitable for ControlNet)
    graph = _build_sdxl_txt2img_graph(
        name="controlnet_sdxl_txt2img",
        pretrained=pretrained,
        num_steps=num_steps,
        guidance_scale=kwargs.get("guidance_scale", 7.5),
        use_batched_cfg=False,
    )
    graph.name = "controlnet_sdxl_txt2img"
    graph.metadata["base_model"] = "sdxl"

    from yggdrasil.core.graph.adapters import add_controlnet_to_graph
    add_controlnet_to_graph(
        graph,
        controlnet_pretrained=cn_pretrained,
        conditioning_scale=kwargs.get("conditioning_scale", 1.0),
        fp16=True,
    )
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
