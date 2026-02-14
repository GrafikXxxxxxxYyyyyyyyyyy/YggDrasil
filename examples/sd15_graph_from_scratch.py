#!/usr/bin/env python3
"""YggDrasil — Сборка графа SD 1.5 полностью вручную.

Показывает как собрать пайплайн блок за блоком, без шаблонов.
Максимальный контроль — настоящий Lego-конструктор.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from yggdrasil.core.graph.graph import ComputeGraph
from yggdrasil.core.graph.subgraph import LoopSubGraph
from yggdrasil.core.block.builder import BlockBuilder


def main():
    print("=== Сборка SD 1.5 вручную из Lego-блоков ===\n")

    pretrained = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    # ==================== 1. Создаём отдельные блоки ====================

    print("1. Создаём блоки...")

    conditioner = BlockBuilder.build({
        "type": "conditioner/clip_text",
        "pretrained": pretrained,
        "tokenizer_subfolder": "tokenizer",
        "text_encoder_subfolder": "text_encoder",
        "max_length": 77,
    })
    print(f"   conditioner: {type(conditioner).__name__}")

    backbone = BlockBuilder.build({
        "type": "backbone/unet2d_condition",
        "pretrained": pretrained,
        "fp16": True,
    })
    print(f"   backbone:    {type(backbone).__name__}")

    guidance = BlockBuilder.build({
        "type": "guidance/cfg",
        "scale": 7.5,
    })
    guidance._backbone_ref = backbone  # для внутреннего dual-pass
    print(f"   guidance:    {type(guidance).__name__} (scale={guidance.scale})")

    solver = BlockBuilder.build({
        "type": "diffusion/solver/ddim",
        "eta": 0.0,
    })
    print(f"   solver:      {type(solver).__name__}")

    codec = BlockBuilder.build({
        "type": "codec/autoencoder_kl",
        "pretrained": pretrained,
        "fp16": True,
        "scaling_factor": 0.18215,
        "latent_channels": 4,
    })
    print(f"   codec:       {type(codec).__name__}")

    # ==================== 2. Собираем inner graph (один шаг деноизинга) ====================

    print("\n2. Собираем inner graph (один шаг)...")

    step = ComputeGraph("denoise_step")
    step.add_node("backbone", backbone)
    step.add_node("guidance", guidance)
    step.add_node("solver", solver)

    # Рёбра: backbone → guidance → solver
    step.connect("backbone", "output", "guidance", "model_output")
    step.connect("guidance", "guided_output", "solver", "model_output")

    # Fan-out: latents → backbone.x + solver.current_latents + guidance.x
    step.expose_input("latents", "backbone", "x")
    step.expose_input("latents", "solver", "current_latents")
    step.expose_input("latents", "guidance", "x")

    # Fan-out: timestep → backbone.timestep + solver.timestep + guidance.t
    step.expose_input("timestep", "backbone", "timestep")
    step.expose_input("timestep", "solver", "timestep")
    step.expose_input("timestep", "guidance", "t")

    # Fan-out: condition → backbone.condition + guidance.condition
    step.expose_input("condition", "backbone", "condition")
    step.expose_input("condition", "guidance", "condition")

    # next_timestep → solver
    step.expose_input("next_timestep", "solver", "next_timestep")

    # Выходы
    step.expose_output("next_latents", "solver", "next_latents")
    step.expose_output("latents", "solver", "next_latents")

    errors = step.validate()
    print(f"   Валидация: {'OK' if not errors else errors}")
    print(f"   Узлы: {list(step.nodes.keys())}")
    print(f"   Входы: {list(step.graph_inputs.keys())}")

    # ==================== 3. Оборачиваем в LoopSubGraph ====================

    print("\n3. Создаём LoopSubGraph (28 шагов)...")

    loop = LoopSubGraph.create(
        inner_graph=step,
        num_iterations=28,
        carry_vars=["latents"],
        show_progress=True,
    )
    print(f"   {type(loop).__name__} x {loop.num_iterations} итераций")

    # ==================== 4. Собираем внешний граф ====================

    print("\n4. Собираем полный граф...")

    graph = ComputeGraph("sd15_manual")
    graph.add_node("conditioner", conditioner)
    graph.add_node("denoise_loop", loop)
    graph.add_node("codec", codec)

    # prompt → conditioner → loop → codec
    graph.expose_input("prompt", "conditioner", "raw_condition")
    graph.connect("conditioner", "embedding", "denoise_loop", "condition")
    graph.expose_input("latents", "denoise_loop", "initial_latents")
    graph.connect("denoise_loop", "latents", "codec", "latent")

    # Выходы
    graph.expose_output("decoded", "codec", "decoded")
    graph.expose_output("latents", "denoise_loop", "latents")

    errors = graph.validate()
    print(f"   Валидация: {'OK' if not errors else errors}")
    print(f"   Полный граф: {graph}")

    # ==================== 5. Визуализация ====================

    print("\n5. Структура графа:")
    print(f"   Узлы ({len(graph.nodes)}):")
    for name, block in graph.nodes.items():
        btype = getattr(block, "block_type", type(block).__name__)
        print(f"     {name}: {btype}")

    print(f"\n   Рёбра ({len(graph.edges)}):")
    for e in graph.edges:
        print(f"     {e.src_node}.{e.src_port} → {e.dst_node}.{e.dst_port}")

    print(f"\n   Входы:")
    for name, targets in graph.graph_inputs.items():
        for node, port in targets:
            print(f"     {name} → {node}.{port}")

    print(f"\n   Выходы:")
    for name, (node, port) in graph.graph_outputs.items():
        print(f"     {name} ← {node}.{port}")

    # ==================== 6. Mermaid ====================

    print(f"\n6. Mermaid-диаграмма:\n{graph.visualize()}")

    # ==================== 7. Сохранение в YAML ====================

    yaml_path = Path("output") / "sd15_manual_graph.yaml"
    graph.to_yaml(yaml_path)
    print(f"\n7. Граф сохранён в {yaml_path}")

    # ==================== 8. Запуск (если есть GPU) ====================

    print("\n8. Для запуска:")
    print('   graph.to("cuda")  # перенос всего графа одной строкой')
    print('   outputs = graph.execute(prompt="your prompt", num_steps=28, seed=42)')
    print("   image = outputs['decoded']")


if __name__ == "__main__":
    main()
