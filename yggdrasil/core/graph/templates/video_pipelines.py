# yggdrasil/core/graph/templates/video_pipelines.py
"""Блочные (Lego) шаблоны видео-диффузии и анимации изображений.

Все пайплайны собираются из одних и тех же блоков: conditioner, backbone, guidance,
solver, codec; цикл деноизинга — LoopSubGraph(denoise_step). Параллель diffusers v0.36.0:
AnimateDiff, CogVideoX, EasyAnimate, HunyuanVideo, I2VGen-XL, Latte, Sana Video,
Stable Video Diffusion, Text2Video, WAN, SkyReels, LTX и др.

Структура: prompt -> conditioner [+ negative] -> denoise_loop -> codec -> decoded (video).
Для img2vid дополнительно: source_image -> codec.encode или отдельный вход.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasil.core.graph.graph import ComputeGraph
from yggdrasil.core.graph.templates import register_template
from yggdrasil.core.graph.templates.image_pipelines import _build_denoise_step
from yggdrasil.core.graph.adapters import add_optional_adapters_to_graph


_DEFAULT_SCHEDULE = {"num_train_timesteps": 1000}
_DEFAULT_SOLVER_DDIM = {
    "type": "diffusion/solver/ddim",
    "eta": 0.0,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "num_train_timesteps": 1000,
    "steps_offset": 1,
}


def _build_video_txt2vid_graph(
    name: str,
    backbone_config: Dict[str, Any],
    codec_config: Dict[str, Any],
    conditioner_configs: List[Dict[str, Any]],
    guidance_config: Dict[str, Any],
    solver_config: Dict[str, Any],
    schedule_config: Dict[str, Any],
    *,
    num_steps: int = 50,
    default_width: int = 512,
    default_height: int = 512,
) -> ComputeGraph:
    """Собрать text-to-video граф из блоков (Lego). Без скрытых зависимостей."""
    from yggdrasil.core.block.builder import BlockBuilder
    from yggdrasil.core.graph.subgraph import LoopSubGraph

    use_cfg = guidance_config.get("scale", 7.5) > 1.0
    graph = ComputeGraph(name)
    graph.metadata = {
        "modality": "video",
        "default_guidance_scale": guidance_config.get("scale", 7.5),
        "default_num_steps": num_steps,
        "default_width": default_width,
        "default_height": default_height,
        "num_frames": 16,
        "latent_channels": codec_config.get("latent_channels", 4),
        "spatial_scale_factor": codec_config.get("spatial_scale_factor", 8),
        "init_noise_sigma": solver_config.get("init_noise_sigma", 1.0),
    }

    backbone = BlockBuilder.build(backbone_config)
    codec = BlockBuilder.build(codec_config)
    guidance = BlockBuilder.build(guidance_config)
    solver = BlockBuilder.build(solver_config)
    step_graph = _build_denoise_step(backbone, guidance, solver, use_cfg=use_cfg)
    num_train_t = int(schedule_config.get("num_train_timesteps", 1000))
    steps_offset = int(solver_config.get("steps_offset", 0))
    loop = LoopSubGraph.create(
        inner_graph=step_graph,
        num_iterations=num_steps,
        carry_vars=["latents"],
        num_train_timesteps=num_train_t,
        timestep_spacing="leading",
        steps_offset=steps_offset,
    )
    loop.show_progress = True

    first_cond = None
    for i, cond_config in enumerate(conditioner_configs):
        cond = BlockBuilder.build(cond_config)
        if i == 0:
            first_cond = cond
        graph.add_node(f"conditioner_{i}", cond)
    if use_cfg:
        graph.add_node("conditioner_negative", first_cond)

    graph.add_node("denoise_loop", loop)
    graph.add_node("codec", codec)

    graph.expose_input("prompt", "conditioner_0", "raw_condition")
    for i in range(1, len(conditioner_configs)):
        graph.expose_input(f"prompt_{i}", f"conditioner_{i}", "raw_condition")
    graph.connect("conditioner_0", "embedding", "denoise_loop", "condition")
    if use_cfg:
        graph.expose_input("negative_prompt", "conditioner_negative", "raw_condition")
        graph.connect("conditioner_negative", "embedding", "denoise_loop", "uncond")

    graph.expose_input("latents", "denoise_loop", "initial_latents")
    graph.expose_input("timesteps", "denoise_loop", "timesteps")
    graph.connect("denoise_loop", "latents", "codec", "latent")
    graph.expose_output("decoded", "codec", "decoded")
    graph.expose_output("latents", "denoise_loop", "latents")
    # Optional adapters: ControlNet, T2I-Adapter, IP-Adapter (only when backbone supports adapter_features)
    add_optional_adapters_to_graph(graph, controlnet=True, t2i_adapter=True, ip_adapter=True)
    return graph


def _build_video_img2vid_graph(
    name: str,
    backbone_config: Dict[str, Any],
    codec_config: Dict[str, Any],
    conditioner_config: Dict[str, Any],
    guidance_config: Dict[str, Any],
    solver_config: Dict[str, Any],
    schedule_config: Dict[str, Any],
    *,
    num_steps: int = 25,
    default_width: int = 1024,
    default_height: int = 576,
) -> ComputeGraph:
    """Image-to-video: опциональный source_image (первый кадр или условие)."""
    graph = _build_video_txt2vid_graph(
        name=name,
        backbone_config=backbone_config,
        codec_config=codec_config,
        conditioner_configs=[conditioner_config],
        guidance_config=guidance_config,
        solver_config=solver_config,
        schedule_config=schedule_config,
        num_steps=num_steps,
        default_width=default_width,
        default_height=default_height,
    )
    graph.expose_input("source_image", "codec", "pixel_data")
    return graph


# ==================== TEXT-TO-VIDEO ====================

@register_template("animatediff_txt2vid")
def animatediff_txt2vid(**kwargs) -> ComputeGraph:
    """AnimateDiff: анимация изображений / text-to-video (SD 1.5 + motion modules). Diffusers: AnimateDiffPipeline."""
    pretrained = kwargs.get("pretrained", "runwayml/stable-diffusion-v1-5")
    graph = _build_video_txt2vid_graph(
        name="animatediff_txt2vid",
        backbone_config={"type": "backbone/unet3d_condition", "pretrained": pretrained, "fp16": True},
        codec_config={"type": "codec/autoencoder_kl", "pretrained": pretrained, "fp16": True, "scaling_factor": 0.18215, "latent_channels": 4, "spatial_scale_factor": 8},
        conditioner_configs=[{"type": "conditioner/clip_text", "pretrained": pretrained, "tokenizer_subfolder": "tokenizer", "text_encoder_subfolder": "text_encoder", "max_length": 77}],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 7.5)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 25),
        default_width=512,
        default_height=512,
    )
    graph.metadata["num_frames"] = kwargs.get("num_frames", 16)
    return graph


@register_template("cogvideox_txt2vid")
def cogvideox_txt2vid(**kwargs) -> ComputeGraph:
    """CogVideoX: text-to-video. Diffusers: CogVideoXPipeline."""
    return _build_video_txt2vid_graph(
        name="cogvideox_txt2vid",
        backbone_config={"type": "backbone/dit", "hidden_dim": 3072, "num_layers": 30, "num_heads": 24, "in_channels": 16, "patch_size": 2},
        codec_config={"type": "codec/autoencoder_kl", "latent_channels": 16, "spatial_scale_factor": 2},
        conditioner_configs=[{"type": "conditioner/t5_text", "pretrained": "google/t5-v1_1-xxl", "max_length": 226}],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 6.0)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 50),
        default_width=480,
        default_height=320,
    )


@register_template("latte_txt2vid")
def latte_txt2vid(**kwargs) -> ComputeGraph:
    """Latte: text-to-video (DiT-based). Diffusers: LattePipeline."""
    return _build_video_txt2vid_graph(
        name="latte_txt2vid",
        backbone_config={"type": "backbone/dit", "pretrained": kwargs.get("pretrained", "maxwell-chen/Latte"), "hidden_dim": 1024, "num_layers": 24, "num_heads": 16, "in_channels": 16, "patch_size": 1},
        codec_config={"type": "codec/autoencoder_kl", "latent_channels": 16},
        conditioner_configs=[{"type": "conditioner/t5_text", "pretrained": "google/t5-v1_1-xxl", "max_length": 512}],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 6.0)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 50),
    )


@register_template("easyanimate_txt2vid")
def easyanimate_txt2vid(**kwargs) -> ComputeGraph:
    """EasyAnimate: text-to-video. Diffusers: EasyAnimatePipeline."""
    pretrained = kwargs.get("pretrained", "hnanaE/EasyAnimate")
    return _build_video_txt2vid_graph(
        name="easyanimate_txt2vid",
        backbone_config={"type": "backbone/dit", "pretrained": pretrained, "hidden_dim": 3072, "num_layers": 30, "num_heads": 24, "in_channels": 16, "patch_size": 2},
        codec_config={"type": "codec/autoencoder_kl", "latent_channels": 16},
        conditioner_configs=[{"type": "conditioner/t5_text", "pretrained": "google/t5-v1_1-xxl", "max_length": 512}],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 6.0)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 50),
    )


@register_template("text_to_video_sd")
def text_to_video_sd(**kwargs) -> ComputeGraph:
    """Text-to-Video (SD-based). Diffusers: TextToVideoSDPipeline."""
    pretrained = kwargs.get("pretrained", "runwayml/stable-diffusion-v1-5")
    return _build_video_txt2vid_graph(
        name="text_to_video_sd",
        backbone_config={"type": "backbone/unet3d_condition", "pretrained": pretrained, "fp16": True},
        codec_config={"type": "codec/autoencoder_kl", "pretrained": pretrained, "fp16": True, "latent_channels": 4, "spatial_scale_factor": 8},
        conditioner_configs=[{"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14", "max_length": 77}],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 7.5)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 50),
    )


@register_template("wan_txt2vid")
def wan_txt2vid(**kwargs) -> ComputeGraph:
    """WAN: text-to-video. Diffusers: WanPipeline."""
    return _build_video_txt2vid_graph(
        name="wan_txt2vid",
        backbone_config={"type": "backbone/wan_transformer", "pretrained": kwargs.get("pretrained", "ali-vilab/wan2.1")},
        codec_config={"type": "codec/wan_vae", "latent_channels": 16},
        conditioner_configs=[{"type": "conditioner/t5_text", "pretrained": "google/t5-v1_1-xxl", "max_length": 512}],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 6.0)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 50),
    )


@register_template("sana_video_txt2vid")
def sana_video_txt2vid(**kwargs) -> ComputeGraph:
    """Sana Video: text-to-video. Diffusers: SanaVideoPipeline."""
    return _build_video_txt2vid_graph(
        name="sana_video_txt2vid",
        backbone_config={"type": "backbone/dit", "pretrained": kwargs.get("pretrained", "sana-dev/Sana-Video"), "hidden_dim": 4096, "num_layers": 24, "num_heads": 16, "in_channels": 16, "patch_size": 1},
        codec_config={"type": "codec/autoencoder_kl", "latent_channels": 16},
        conditioner_configs=[{"type": "conditioner/t5_text", "pretrained": "google/t5-v1_1-xxl", "max_length": 512}],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 6.0)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 50),
    )


@register_template("hunyuan_video_txt2vid")
def hunyuan_video_txt2vid(**kwargs) -> ComputeGraph:
    """HunyuanVideo: text-to-video. Diffusers: HunyuanVideoPipeline."""
    return _build_video_txt2vid_graph(
        name="hunyuan_video_txt2vid",
        backbone_config={"type": "backbone/unet3d_condition", "pretrained": kwargs.get("pretrained", "tencent-hunyuan/HunyuanVideo")},
        codec_config={"type": "codec/autoencoder_kl", "latent_channels": 16},
        conditioner_configs=[{"type": "conditioner/t5_text", "pretrained": "google/t5-v1_1-xxl", "max_length": 512}],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 6.0)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 50),
    )


@register_template("ltx_txt2vid")
def ltx_txt2vid(**kwargs) -> ComputeGraph:
    """LTX Video: text-to-video. Diffusers: LTXPipeline."""
    return _build_video_txt2vid_graph(
        name="ltx_txt2vid",
        backbone_config={"type": "backbone/dit", "pretrained": kwargs.get("pretrained", "Lightricks/LTX-Video")},
        codec_config={"type": "codec/autoencoder_kl", "latent_channels": 16},
        conditioner_configs=[{"type": "conditioner/t5_text", "pretrained": "google/t5-v1_1-xxl", "max_length": 512}],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 6.0)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 50),
    )


# ==================== IMAGE-TO-VIDEO ====================

@register_template("stable_video_diffusion")
def stable_video_diffusion(**kwargs) -> ComputeGraph:
    """Stable Video Diffusion: image-to-video. Diffusers: StableVideoDiffusionPipeline."""
    pretrained = kwargs.get("pretrained", "stabilityai/stable-video-diffusion-img2vid-xt")
    return _build_video_img2vid_graph(
        name="stable_video_diffusion",
        backbone_config={"type": "backbone/unet3d_condition", "pretrained": pretrained, "fp16": True},
        codec_config={"type": "codec/autoencoder_kl", "pretrained": pretrained, "fp16": True, "latent_channels": 4, "spatial_scale_factor": 8},
        conditioner_config={"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14"},
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 3.0)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 25),
        default_width=1024,
        default_height=576,
    )


@register_template("i2vgen_xl")
def i2vgen_xl(**kwargs) -> ComputeGraph:
    """I2VGen-XL: image-to-video. Diffusers: I2VGenXLPipeline."""
    pretrained = kwargs.get("pretrained", "ali-vilab/i2vgen-xl")
    return _build_video_img2vid_graph(
        name="i2vgen_xl",
        backbone_config={"type": "backbone/unet3d_condition", "pretrained": pretrained, "fp16": True},
        codec_config={"type": "codec/autoencoder_kl", "latent_channels": 4, "spatial_scale_factor": 8},
        conditioner_config={"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14"},
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 7.5)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 50),
    )


@register_template("cogvideox_img2vid")
def cogvideox_img2vid(**kwargs) -> ComputeGraph:
    """CogVideoX image-to-video. Diffusers: CogVideoXImageToVideoPipeline."""
    graph = _build_video_txt2vid_graph(
        name="cogvideox_img2vid",
        backbone_config={"type": "backbone/dit", "hidden_dim": 3072, "num_layers": 30, "num_heads": 24, "in_channels": 16, "patch_size": 2},
        codec_config={"type": "codec/autoencoder_kl", "latent_channels": 16},
        conditioner_configs=[{"type": "conditioner/t5_text", "pretrained": "google/t5-v1_1-xxl", "max_length": 226}],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 6.0)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 50),
    )
    graph.expose_input("source_image", "codec", "pixel_data")
    return graph


@register_template("wan_img2vid")
def wan_img2vid(**kwargs) -> ComputeGraph:
    """WAN image-to-video. Diffusers: WanImageToVideoPipeline."""
    graph = _build_video_txt2vid_graph(
        name="wan_img2vid",
        backbone_config={"type": "backbone/wan_transformer", "pretrained": kwargs.get("pretrained", "ali-vilab/wan2.1")},
        codec_config={"type": "codec/wan_vae", "latent_channels": 16},
        conditioner_configs=[{"type": "conditioner/t5_text", "pretrained": "google/t5-v1_1-xxl", "max_length": 512}],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 6.0)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 50),
    )
    graph.expose_input("source_image", "codec", "pixel_data")
    return graph


@register_template("wan_animate")
def wan_animate(**kwargs) -> ComputeGraph:
    """WAN Animate: анимация изображений. Diffusers: WanAnimatePipeline."""
    graph = _build_video_txt2vid_graph(
        name="wan_animate",
        backbone_config={"type": "backbone/wan_transformer", "pretrained": kwargs.get("pretrained", "ali-vilab/wan2.1-animate")},
        codec_config={"type": "codec/wan_vae", "latent_channels": 16},
        conditioner_configs=[{"type": "conditioner/t5_text", "pretrained": "google/t5-v1_1-xxl", "max_length": 512}],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 6.0)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 50),
    )
    graph.expose_input("source_image", "codec", "pixel_data")
    return graph


# ==================== REGISTRY ====================

VIDEO_TEMPLATES = [
    "animatediff_txt2vid",
    "cogvideox_txt2vid",
    "latte_txt2vid",
    "easyanimate_txt2vid",
    "text_to_video_sd",
    "wan_txt2vid",
    "sana_video_txt2vid",
    "hunyuan_video_txt2vid",
    "ltx_txt2vid",
    "stable_video_diffusion",
    "i2vgen_xl",
    "cogvideox_img2vid",
    "wan_img2vid",
    "wan_animate",
]


def list_video_templates() -> Dict[str, str]:
    """Имена и краткие описания всех видео-шаблонов."""
    return {
        "animatediff_txt2vid": "AnimateDiff — анимация / text-to-video (SD 1.5)",
        "cogvideox_txt2vid": "CogVideoX — text-to-video",
        "latte_txt2vid": "Latte — text-to-video (DiT)",
        "easyanimate_txt2vid": "EasyAnimate — text-to-video",
        "text_to_video_sd": "Text2Video SD — text-to-video (SD-based)",
        "wan_txt2vid": "WAN — text-to-video",
        "sana_video_txt2vid": "Sana Video — text-to-video",
        "hunyuan_video_txt2vid": "HunyuanVideo — text-to-video",
        "ltx_txt2vid": "LTX Video — text-to-video",
        "stable_video_diffusion": "Stable Video Diffusion — image-to-video",
        "i2vgen_xl": "I2VGen-XL — image-to-video",
        "cogvideox_img2vid": "CogVideoX — image-to-video",
        "wan_img2vid": "WAN — image-to-video",
        "wan_animate": "WAN Animate — анимация изображений",
    }
