# yggdrasil/core/graph/templates/animatediff_extensions.py
"""Расширенные шаблоны AnimateDiff в Lego-стиле: I2V, V2V, совместимость с любыми адаптерами.

Все режимы собираются из тех же блоков, что и animatediff_txt2vid; добавляются только
входы source_image / source_video и опция strength для начальных латентов.
"""
from __future__ import annotations

from typing import Any, Dict

from yggdrasil.core.graph.graph import ComputeGraph
from yggdrasil.core.graph.templates import register_template
from yggdrasil.core.graph.templates.video_pipelines import (
    _build_video_txt2vid_graph,
    _DEFAULT_SCHEDULE,
    _DEFAULT_SOLVER_DDIM,
)


@register_template("animatediff_i2v")
def animatediff_i2v(**kwargs) -> ComputeGraph:
    """AnimateDiff Image-to-Video: анимация статичного изображения.

    Тот же граф, что animatediff_txt2vid, с явным входом source_image.
    При вызове InferencePipeline.__call__(source_image=..., strength=0.7) начальные
    латенты строятся из кодированного изображения + шум (I2V).
    """
    pretrained = kwargs.get("pretrained", "runwayml/stable-diffusion-v1-5")
    graph = _build_video_txt2vid_graph(
        name="animatediff_i2v",
        backbone_config={"type": "backbone/unet3d_condition", "pretrained": pretrained, "fp16": True},
        codec_config={
            "type": "codec/autoencoder_kl",
            "pretrained": pretrained,
            "fp16": True,
            "scaling_factor": 0.18215,
            "latent_channels": 4,
            "spatial_scale_factor": 8,
        },
        conditioner_configs=[
            {
                "type": "conditioner/clip_text",
                "pretrained": pretrained,
                "tokenizer_subfolder": "tokenizer",
                "text_encoder_subfolder": "text_encoder",
                "max_length": 77,
            }
        ],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 7.5)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 25),
        default_width=512,
        default_height=512,
    )
    graph.metadata["num_frames"] = kwargs.get("num_frames", 16)
    graph.metadata["supports_i2v"] = True
    return graph


@register_template("animatediff_v2v")
def animatediff_v2v(**kwargs) -> ComputeGraph:
    """AnimateDiff Video-to-Video: модификация существующего видео.

    Тот же граф, что animatediff_txt2vid; при вызове с source_video и strength
    InferencePipeline строит начальные латенты из кодированного видео и передаёт их как latents.
    """
    pretrained = kwargs.get("pretrained", "runwayml/stable-diffusion-v1-5")
    graph = _build_video_txt2vid_graph(
        name="animatediff_v2v",
        backbone_config={"type": "backbone/unet3d_condition", "pretrained": pretrained, "fp16": True},
        codec_config={
            "type": "codec/autoencoder_kl",
            "pretrained": pretrained,
            "fp16": True,
            "scaling_factor": 0.18215,
            "latent_channels": 4,
            "spatial_scale_factor": 8,
        },
        conditioner_configs=[
            {
                "type": "conditioner/clip_text",
                "pretrained": pretrained,
                "tokenizer_subfolder": "tokenizer",
                "text_encoder_subfolder": "text_encoder",
                "max_length": 77,
            }
        ],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 7.5)},
        solver_config={**_DEFAULT_SOLVER_DDIM},
        schedule_config={**_DEFAULT_SCHEDULE},
        num_steps=kwargs.get("num_steps", 25),
        default_width=512,
        default_height=512,
    )
    graph.metadata["num_frames"] = kwargs.get("num_frames", 16)
    graph.metadata["supports_v2v"] = True
    return graph
