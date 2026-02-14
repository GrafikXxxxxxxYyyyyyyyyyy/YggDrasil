#!/usr/bin/env python3
"""Замена solver (шаг диффузии) без пересборки графа.

Берём граф из шаблона, находим внутренний граф цикла деноайзинга и
заменяем узел "solver" на новый DDIM с другими параметрами (например eta).
Демонстрирует graph.replace_node() и вложенную структуру LoopSubGraph.

Запуск:
    python examples/sd15/05_replace_solver.py
"""
import torch
from yggdrasil.pipeline import Pipeline
from yggdrasil.core.block.builder import BlockBuilder


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    pipe = Pipeline.from_template("sd15_txt2img", device=device)

    # Внутренний граф цикла — узел denoise_loop это LoopSubGraph с полем .graph
    loop = pipe.graph.nodes["denoise_loop"]
    inner = loop.graph

    # Новый solver с другим eta (больше стохастичности)
    new_solver = BlockBuilder.build({
        "type": "diffusion/solver/ddim",
        "eta": 0.5,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "num_train_timesteps": 1000,
        "clip_sample_range": 1.0,
        "steps_offset": 1,
    })
    inner.replace_node("solver", new_solver)
    print("Solver заменён на DDIM с eta=0.5")

    out = pipe("a cloudy sky", num_steps=28, seed=42)
    if out.images:
        out.images[0].save("output_replaced_solver.png")
        print("Сохранено: output_replaced_solver.png")


if __name__ == "__main__":
    main()
