# yggdrasil/core/graph/templates/audio_pipelines.py
"""Блочные (Lego) шаблоны аудио-диффузии — единая система для генерации звука.

Все пайплайны собираются из одних и тех же блоков: conditioner, backbone, guidance,
solver, codec; цикл деноизинга — LoopSubGraph(denoise_step). Полная паритет с
diffusers: AudioLDM, AudioLDM2, MusicLDM, Stable Audio, Dance Diffusion, Aura Flow.

Структура (как в image): prompt -> conditioner [+ negative] -> denoise_loop -> codec -> decoded.
"""
from __future__ import annotations

from typing import Any, Dict

from yggdrasil.core.graph.graph import ComputeGraph
from yggdrasil.core.graph.templates import register_template
from yggdrasil.core.graph.templates.image_pipelines import _build_denoise_step
from yggdrasil.core.graph.adapters import add_optional_adapters_to_graph


# ==================== HELPER: BUILD FULL AUDIO GRAPH ====================

def _build_audio_txt2audio_graph(
    name: str,
    backbone_config: Dict[str, Any],
    codec_config: Dict[str, Any],
    conditioner_config: Dict[str, Any],
    guidance_config: Dict[str, Any],
    solver_config: Dict[str, Any],
    schedule_config: Dict[str, Any],
    *,
    num_steps: int = 50,
    default_audio_latent_height: int = 256,
    default_audio_latent_width: int = 16,
) -> ComputeGraph:
    """Собрать полный text-to-audio граф из блоков (Lego).

    Структура:
        prompt ----------> conditioner_0 (embedding) ---\
        negative_prompt -> conditioner_negative -------+-> denoise_loop -> codec -> decoded
        latents (noise) -> denoise_loop.initial_latents -/

    Все блоки задаются конфигами; граф не содержит скрытых зависимостей.
    """
    from yggdrasil.core.block.builder import BlockBuilder
    from yggdrasil.core.graph.subgraph import LoopSubGraph

    use_cfg = guidance_config.get("scale", 2.5) > 1.0

    graph = ComputeGraph(name)
    graph.metadata = {
        "modality": "audio",
        "default_guidance_scale": guidance_config.get("scale", 2.5),
        "default_num_steps": num_steps,
        "latent_channels": codec_config.get("latent_channels", 8),
        "spatial_scale_factor": codec_config.get("spatial_scale_factor", 1),
        "default_audio_latent_height": default_audio_latent_height,
        "default_audio_latent_width": default_audio_latent_width,
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

    cond = BlockBuilder.build(conditioner_config)
    graph.add_node("conditioner_0", cond)
    if use_cfg:
        cond_neg = BlockBuilder.build(conditioner_config)
        graph.add_node("conditioner_negative", cond_neg)

    graph.add_node("denoise_loop", loop)
    graph.add_node("codec", codec)

    graph.expose_input("prompt", "conditioner_0", "raw_condition")
    if use_cfg:
        graph.expose_input("negative_prompt", "conditioner_negative", "raw_condition")
        graph.connect("conditioner_0", "embedding", "denoise_loop", "condition")
        graph.connect("conditioner_negative", "embedding", "denoise_loop", "uncond")
    else:
        graph.connect("conditioner_0", "embedding", "denoise_loop", "condition")

    graph.expose_input("latents", "denoise_loop", "initial_latents")
    graph.connect("denoise_loop", "latents", "codec", "latent")

    graph.expose_output("decoded", "codec", "decoded")
    graph.expose_output("latents", "denoise_loop", "latents")

    # Optional adapters: ControlNet, T2I-Adapter, IP-Adapter (only when backbone supports adapter_features)
    add_optional_adapters_to_graph(graph, controlnet=True, t2i_adapter=True, ip_adapter=True)

    return graph


# ==================== AUDIO PIPELINE TEMPLATES (diffusers parity) ====================

# Общие конфиги для переиспользования
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


@register_template("audioldm_txt2audio")
def audioldm_txt2audio(**kwargs) -> ComputeGraph:
    """AudioLDM: text-to-audio (CVSSP). Diffusers: AudioLDMPipeline."""
    pretrained = kwargs.get("pretrained", "cvssp/audioldm-l-full")
    return _build_audio_txt2audio_graph(
        name="audioldm_txt2audio",
        backbone_config={"type": "backbone/unet2d_condition", "pretrained": pretrained, "fp16": False},
        codec_config={"type": "codec/autoencoder_kl", "pretrained": pretrained, "latent_channels": 8, "fp16": True},
        conditioner_config={"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14"},
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 2.5)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 200),
    )


@register_template("audioldm2_txt2audio")
def audioldm2_txt2audio(**kwargs) -> ComputeGraph:
    """AudioLDM 2: text-to-audio. Diffusers: AudioLDM2Pipeline."""
    pretrained = kwargs.get("pretrained", "cvssp/audioldm2")
    return _build_audio_txt2audio_graph(
        name="audioldm2_txt2audio",
        backbone_config={"type": "backbone/unet2d_condition", "pretrained": pretrained, "fp16": False},
        codec_config={"type": "codec/autoencoder_kl", "pretrained": pretrained, "latent_channels": 8, "fp16": True},
        conditioner_config={"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14"},
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 3.5)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 200),
    )


@register_template("musicldm_txt2audio")
def musicldm_txt2audio(**kwargs) -> ComputeGraph:
    """MusicLDM: text-to-music (UCSD). Diffusers: MusicLDMPipeline. Conditioner: CLAP."""
    pretrained = kwargs.get("pretrained", "ucsd-reach/musicldm")
    return _build_audio_txt2audio_graph(
        name="musicldm_txt2audio",
        backbone_config={"type": "backbone/unet2d_condition", "pretrained": pretrained, "fp16": True},
        codec_config={"type": "codec/autoencoder_kl", "pretrained": pretrained, "latent_channels": 8, "fp16": True},
        conditioner_config={"type": "conditioner/clap", "pretrained": "laion/clap-htsat-unfused"},
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 2.5)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 100),
    )


@register_template("stable_audio")
def stable_audio(**kwargs) -> ComputeGraph:
    """Stable Audio: text-to-audio (Stability AI). Diffusers: StableAudioPipeline.
    Лучше грузить через DiffusersBridge.from_pretrained('stabilityai/stable-audio-open-1.0')."""
    return _build_audio_txt2audio_graph(
        name="stable_audio",
        backbone_config={
            "type": "backbone/dit",
            "hidden_dim": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "in_channels": 64,
            "patch_size": 1,
        },
        codec_config={"type": "codec/autoencoder_kl", "latent_channels": 64, "fp16": True},
        conditioner_config={"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14"},
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 7.0)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 100),
        default_audio_latent_height=256,
        default_audio_latent_width=16,
    )


@register_template("dance_diffusion_audio")
def dance_diffusion_audio(**kwargs) -> ComputeGraph:
    """Dance Diffusion: unconditional или text-conditioned audio. Diffusers: DanceDiffusionPipeline.
    Внимание: чекпоинт harmonai/maestro-150k использует UNet1D (DownBlock1DNoSkip); загрузка через
    наш backbone/unet2d_condition может падать. Для полной поддержки нужен отдельный 1D-блок."""
    pretrained = kwargs.get("pretrained", "harmonai/maestro-150k")
    return _build_audio_txt2audio_graph(
        name="dance_diffusion_audio",
        backbone_config={"type": "backbone/unet2d_condition", "pretrained": pretrained, "fp16": True},
        codec_config={"type": "codec/autoencoder_kl", "latent_channels": 1, "fp16": True},
        conditioner_config={"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14"},
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 1.0)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 100),
    )


@register_template("aura_flow_audio")
def aura_flow_audio(**kwargs) -> ComputeGraph:
    """Aura Flow: voice/TTS-ориентированный пайплайн. Diffusers: AuraFlowPipeline.
    Репозиторий black-forest-labs/AuraFlow может быть недоступен на HF; укажите pretrained=... при наличии."""
    pretrained = kwargs.get("pretrained", "black-forest-labs/AuraFlow")
    return _build_audio_txt2audio_graph(
        name="aura_flow_audio",
        backbone_config={"type": "backbone/unet2d_condition", "pretrained": pretrained, "fp16": True},
        codec_config={"type": "codec/autoencoder_kl", "pretrained": pretrained, "latent_channels": 8, "fp16": True},
        conditioner_config={"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14"},
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 3.5)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 50),
    )


# ==================== REGISTRY: список шаблонов для UI и from_pretrained ====================

AUDIO_TEMPLATES = [
    "audioldm_txt2audio",
    "audioldm2_txt2audio",
    "musicldm_txt2audio",
    "stable_audio",
    "dance_diffusion_audio",
    "aura_flow_audio",
]


def list_audio_templates() -> Dict[str, str]:
    """Имена и краткие описания всех аудио-шаблонов (для единой системы генерации звука)."""
    return {
        "audioldm_txt2audio": "AudioLDM — text-to-audio (cvssp/audioldm-l-full)",
        "audioldm2_txt2audio": "AudioLDM 2 — text-to-audio (cvssp/audioldm2)",
        "musicldm_txt2audio": "MusicLDM — text-to-music (ucsd-reach/musicldm)",
        "stable_audio": "Stable Audio — text-to-audio (Stability AI)",
        "dance_diffusion_audio": "Dance Diffusion — audio (harmonai/maestro-150k)",
        "aura_flow_audio": "Aura Flow — voice/TTS (black-forest-labs/AuraFlow)",
    }
