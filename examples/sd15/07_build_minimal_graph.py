#!/usr/bin/env python3
"""Сборка минимального txt2img графа из блоков вручную.

Без использования готового шаблона: создаём блоки через BlockBuilder,
собираем шаг деноайзинга (backbone -> guidance -> solver), оборачиваем
в LoopSubGraph, подключаем conditioner и codec. Показывает полную
архитектуру SD 1.5 как графа блоков.

Запуск:
    python examples/sd15/07_build_minimal_graph.py
"""
import torch
from yggdrasil.core.graph.graph import ComputeGraph
from yggdrasil.core.graph.templates.image_pipelines import _build_denoise_step, _build_txt2img_graph
from yggdrasil.core.block.builder import BlockBuilder
from yggdrasil.core.graph.subgraph import LoopSubGraph
from yggdrasil.pipeline import Pipeline


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    pretrained = "runwayml/stable-diffusion-v1-5"

    # Блоки
    backbone = BlockBuilder.build({"type": "backbone/unet2d_condition", "pretrained": pretrained, "fp16": True})
    guidance = BlockBuilder.build({"type": "guidance/cfg", "scale": 7.5, "guidance_rescale": 0.7})
    solver = BlockBuilder.build({
        "type": "diffusion/solver/ddim",
        "eta": 0.0,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "num_train_timesteps": 1000,
        "clip_sample_range": 1.0,
        "steps_offset": 1,
    })
    conditioner = BlockBuilder.build({
        "type": "conditioner/clip_text",
        "pretrained": pretrained,
        "tokenizer_subfolder": "tokenizer",
        "text_encoder_subfolder": "text_encoder",
        "max_length": 77,
    })
    conditioner_neg = BlockBuilder.build({
        "type": "conditioner/clip_text",
        "pretrained": pretrained,
        "tokenizer_subfolder": "tokenizer",
        "text_encoder_subfolder": "text_encoder",
        "max_length": 77,
    })
    codec = BlockBuilder.build({
        "type": "codec/autoencoder_kl",
        "pretrained": pretrained,
        "fp16": True,
        "scaling_factor": 0.18215,
        "latent_channels": 4,
        "spatial_scale_factor": 8,
    })

    step = _build_denoise_step(backbone, guidance, solver, use_cfg=True)
    loop = LoopSubGraph.create(
        inner_graph=step,
        num_iterations=28,
        carry_vars=["latents"],
        num_train_timesteps=1000,
        timestep_spacing="leading",
        steps_offset=1,
    )

    graph = ComputeGraph("sd15_minimal")
    graph.metadata = {"latent_channels": 4, "spatial_scale_factor": 8, "default_num_steps": 28, "default_width": 512, "default_height": 512}
    graph.add_node("conditioner_0", conditioner)
    graph.add_node("conditioner_negative", conditioner_neg)
    graph.add_node("denoise_loop", loop)
    graph.add_node("codec", codec)

    graph.expose_input("prompt", "conditioner_0", "raw_condition")
    graph.expose_input("negative_prompt", "conditioner_negative", "raw_condition")
    graph.connect("conditioner_0", "embedding", "denoise_loop", "condition")
    graph.connect("conditioner_negative", "embedding", "denoise_loop", "uncond")
    graph.expose_input("latents", "denoise_loop", "initial_latents")
    graph.connect("denoise_loop", "latents", "codec", "latent")
    graph.expose_output("decoded", "codec", "decoded")
    graph.expose_output("latents", "denoise_loop", "latents")

    pipe = Pipeline(graph)
    pipe.graph.to(device)

    out = pipe("a cute cat", negative_prompt="", num_steps=28, seed=42)
    if out.images:
        out.images[0].save("output_minimal_graph.png")
        print("Сохранено: output_minimal_graph.png (граф собран вручную)")


if __name__ == "__main__":
    main()
