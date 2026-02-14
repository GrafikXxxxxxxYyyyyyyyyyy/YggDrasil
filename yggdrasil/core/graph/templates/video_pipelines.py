# yggdrasil/core/graph/templates/video_pipelines.py
"""Graph templates for video diffusion pipelines.

All pipelines use LoopSubGraph for proper denoising.
"""
from __future__ import annotations
from yggdrasil.core.graph.graph import ComputeGraph
from yggdrasil.core.graph.templates import register_template
from yggdrasil.core.graph.templates.image_pipelines import _build_denoise_step


def _build_video_pipeline(name, backbone_config, codec_config, conditioner_config, guidance_scale=7.5, num_steps=50, **kwargs):
    """Generic video pipeline builder with denoising loop."""
    from yggdrasil.core.block.builder import BlockBuilder
    from yggdrasil.core.graph.subgraph import LoopSubGraph
    
    backbone = BlockBuilder.build(backbone_config)
    guidance = BlockBuilder.build({"type": "guidance/cfg", "scale": guidance_scale})
    solver = BlockBuilder.build({"type": "diffusion/solver/ddim"})
    conditioner = BlockBuilder.build(conditioner_config)
    codec = BlockBuilder.build(codec_config)
    
    guidance._backbone_ref = backbone
    
    step = _build_denoise_step(backbone, guidance, solver)
    loop = LoopSubGraph.create(inner_graph=step, num_iterations=num_steps)
    
    graph = ComputeGraph(name)
    graph.add_node("conditioner_0", conditioner)
    graph.add_node("denoise_loop", loop)
    graph.add_node("codec", codec)
    
    graph.expose_input("prompt", "conditioner_0", "raw_condition")
    graph.connect("conditioner_0", "embedding", "denoise_loop", "condition")
    graph.expose_input("latents", "denoise_loop", "initial_latents")
    graph.connect("denoise_loop", "latents", "codec", "latent")
    
    graph.expose_output("decoded", "codec", "decoded")
    graph.expose_output("latents", "denoise_loop", "latents")
    
    return graph


@register_template("animatediff_txt2vid")
def animatediff_txt2vid(**kwargs) -> ComputeGraph:
    """AnimateDiff text-to-video pipeline."""
    pretrained = kwargs.get("pretrained", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    return _build_video_pipeline(
        name="animatediff_txt2vid",
        backbone_config={"type": "backbone/unet3d_condition", "pretrained": pretrained, "fp16": True},
        codec_config={"type": "codec/autoencoder_kl", "pretrained": pretrained, "fp16": True, "scaling_factor": 0.18215, "latent_channels": 4},
        conditioner_config={"type": "conditioner/clip_text", "pretrained": pretrained, "tokenizer_subfolder": "tokenizer", "text_encoder_subfolder": "text_encoder", "max_length": 77},
        guidance_scale=kwargs.get("guidance_scale", 7.5),
        num_steps=kwargs.get("num_steps", 25),
    )


@register_template("cogvideox_txt2vid")
def cogvideox_txt2vid(**kwargs) -> ComputeGraph:
    """CogVideoX text-to-video pipeline."""
    return _build_video_pipeline(
        name="cogvideox_txt2vid",
        backbone_config={"type": "backbone/dit", "hidden_dim": 3072, "num_layers": 30, "num_heads": 24, "in_channels": 16, "patch_size": 2},
        codec_config={"type": "codec/autoencoder_kl", "latent_channels": 16},
        conditioner_config={"type": "conditioner/t5_text", "pretrained": "google/t5-v1_1-xxl", "max_length": 226},
        guidance_scale=kwargs.get("guidance_scale", 6.0),
        num_steps=kwargs.get("num_steps", 50),
    )


@register_template("stable_video_diffusion")
def stable_video_diffusion(**kwargs) -> ComputeGraph:
    """Stable Video Diffusion (image-to-video)."""
    graph = _build_video_pipeline(
        name="stable_video_diffusion",
        backbone_config={"type": "backbone/unet3d_condition", "pretrained": kwargs.get("pretrained", "stabilityai/stable-video-diffusion-img2vid-xt"), "fp16": True},
        codec_config={"type": "codec/autoencoder_kl", "pretrained": kwargs.get("pretrained", "stabilityai/stable-video-diffusion-img2vid-xt"), "fp16": True, "latent_channels": 4},
        conditioner_config={"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14"},
        guidance_scale=kwargs.get("guidance_scale", 3.0),
        num_steps=kwargs.get("num_steps", 25),
    )
    graph.expose_input("source_image", "codec", "pixel_data")
    return graph


@register_template("i2vgen_xl")
def i2vgen_xl(**kwargs) -> ComputeGraph:
    """I2VGen-XL image-to-video."""
    graph = _build_video_pipeline(
        name="i2vgen_xl",
        backbone_config={"type": "backbone/unet3d_condition", "pretrained": kwargs.get("pretrained", "ali-vilab/i2vgen-xl"), "fp16": True},
        codec_config={"type": "codec/autoencoder_kl", "latent_channels": 4},
        conditioner_config={"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14"},
        guidance_scale=kwargs.get("guidance_scale", 7.5),
        num_steps=kwargs.get("num_steps", 50),
    )
    graph.expose_input("source_image", "codec", "pixel_data")
    return graph
