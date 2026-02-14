# yggdrasil/core/graph/templates/audio_pipelines.py
"""Graph templates for audio diffusion pipelines.

All pipelines use LoopSubGraph for proper denoising.
"""
from __future__ import annotations
from yggdrasil.core.graph.graph import ComputeGraph
from yggdrasil.core.graph.templates import register_template
from yggdrasil.core.graph.templates.image_pipelines import _build_denoise_step


def _build_audio_pipeline(name, backbone_config, codec_config, conditioner_config, guidance_scale=2.5, num_steps=50):
    """Generic audio pipeline builder with denoising loop."""
    from yggdrasil.core.block.builder import BlockBuilder
    from yggdrasil.core.graph.subgraph import LoopSubGraph
    
    backbone = BlockBuilder.build(backbone_config)
    guidance = BlockBuilder.build({"type": "guidance/cfg", "scale": guidance_scale})
    solver = BlockBuilder.build({"type": "diffusion/solver/ddim"})
    conditioner = BlockBuilder.build(conditioner_config)
    conditioner_uncond = BlockBuilder.build(conditioner_config)
    codec = BlockBuilder.build(codec_config)

    guidance._backbone_ref = backbone

    step = _build_denoise_step(backbone, guidance, solver, use_cfg=True)
    loop = LoopSubGraph.create(
        inner_graph=step,
        num_iterations=num_steps,
        num_train_timesteps=1000,
        timestep_spacing="leading",
        steps_offset=1,
    )
    loop.show_progress = True

    graph = ComputeGraph(name)
    graph.add_node("conditioner_0", conditioner)
    graph.add_node("conditioner_uncond", conditioner_uncond)
    graph.add_node("denoise_loop", loop)
    graph.add_node("codec", codec)

    graph.expose_input("prompt", "conditioner_0", "raw_condition")
    graph.expose_input("negative_prompt", "conditioner_uncond", "raw_condition")
    graph.connect("conditioner_0", "embedding", "denoise_loop", "condition")
    graph.connect("conditioner_uncond", "embedding", "denoise_loop", "uncond")
    graph.expose_input("latents", "denoise_loop", "initial_latents")
    graph.connect("denoise_loop", "latents", "codec", "latent")
    
    graph.expose_output("decoded", "codec", "decoded")
    graph.expose_output("latents", "denoise_loop", "latents")

    # Metadata for pipeline (noise shape, modality)
    latent_channels = codec_config.get("latent_channels", 8)
    graph.metadata = {
        "modality": "audio",
        "latent_channels": latent_channels,
        "spatial_scale_factor": 1,
        "default_audio_latent_height": 256,
        "default_audio_latent_width": 16,
        "init_noise_sigma": 1.0,
    }
    return graph


@register_template("audioldm_txt2audio")
def audioldm_txt2audio(**kwargs) -> ComputeGraph:
    """AudioLDM text-to-audio."""
    return _build_audio_pipeline(
        name="audioldm_txt2audio",
        backbone_config={"type": "backbone/unet2d_condition", "pretrained": kwargs.get("pretrained", "cvssp/audioldm-l-full"), "fp16": True},
        codec_config={"type": "codec/autoencoder_kl", "latent_channels": 8},
        conditioner_config={"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14"},
        guidance_scale=kwargs.get("guidance_scale", 2.5),
        num_steps=kwargs.get("num_steps", 200),
    )


@register_template("audioldm2_txt2audio")
def audioldm2_txt2audio(**kwargs) -> ComputeGraph:
    """AudioLDM 2 text-to-audio."""
    return _build_audio_pipeline(
        name="audioldm2_txt2audio",
        backbone_config={"type": "backbone/unet2d_condition", "pretrained": kwargs.get("pretrained", "cvssp/audioldm2"), "fp16": True},
        codec_config={"type": "codec/autoencoder_kl", "latent_channels": 8},
        conditioner_config={"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14"},
        guidance_scale=kwargs.get("guidance_scale", 3.5),
        num_steps=kwargs.get("num_steps", 200),
    )


@register_template("stable_audio")
def stable_audio(**kwargs) -> ComputeGraph:
    """Stable Audio text-to-audio (Stability AI). Prefer loading via DiffusersBridge.from_pretrained('stabilityai/stable-audio-open-1.0')."""
    return _build_audio_pipeline(
        name="stable_audio",
        backbone_config={"type": "backbone/dit", "hidden_dim": 1024, "num_layers": 24, "num_heads": 16, "in_channels": 64, "patch_size": 1},
        codec_config={"type": "codec/autoencoder_kl", "latent_channels": 64},
        conditioner_config={"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14"},
        guidance_scale=kwargs.get("guidance_scale", 7.0),
        num_steps=kwargs.get("num_steps", 100),
    )


@register_template("musicldm_txt2audio")
def musicldm_txt2audio(**kwargs) -> ComputeGraph:
    """MusicLDM text-to-music (UCSD). https://huggingface.co/ucsd-reach/musicldm"""
    return _build_audio_pipeline(
        name="musicldm_txt2audio",
        backbone_config={
            "type": "backbone/unet2d_condition",
            "pretrained": kwargs.get("pretrained", "ucsd-reach/musicldm"),
            "fp16": True,
        },
        codec_config={"type": "codec/autoencoder_kl", "pretrained": kwargs.get("pretrained", "ucsd-reach/musicldm"), "latent_channels": 8, "fp16": True},
        conditioner_config={"type": "conditioner/clap", "pretrained": "laion/clap-htsat-unfused"},
        guidance_scale=kwargs.get("guidance_scale", 2.5),
        num_steps=kwargs.get("num_steps", 100),
    )


@register_template("dance_diffusion_audio")
def dance_diffusion_audio(**kwargs) -> ComputeGraph:
    """Dance Diffusion — unconditional or prompt-conditioned audio. Uses UNet1D; best loaded via DiffusersBridge.from_pretrained."""
    # Minimal template: 1D UNet is wrapped as backbone/unet2d_condition by MODEL_MAP when importing
    pretrained = kwargs.get("pretrained", "harmonai/maestro-150k")
    return _build_audio_pipeline(
        name="dance_diffusion_audio",
        backbone_config={
            "type": "backbone/unet2d_condition",
            "pretrained": pretrained,
            "fp16": True,
        },
        codec_config={"type": "codec/autoencoder_kl", "latent_channels": 1},
        conditioner_config={"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14"},
        guidance_scale=kwargs.get("guidance_scale", 1.0),
        num_steps=kwargs.get("num_steps", 100),
    )
