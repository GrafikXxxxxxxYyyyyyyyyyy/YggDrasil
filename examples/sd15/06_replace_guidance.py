#!/usr/bin/env python3
"""Замена блока guidance (CFG) — другой scale или отключение CFG.

Находим узел guidance во внутреннем графе шага и заменяем его на новый
ClassifierFreeGuidance с scale=3.0 или scale=1.0 (без guidance).
Показывает замену компонента, влияющего на силу текстовой conditioning.

Запуск:
    python examples/sd15/06_replace_guidance.py
"""
import torch
from yggdrasil.pipeline import Pipeline
from yggdrasil.core.block.builder import BlockBuilder


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    pipe = Pipeline.from_template("sd15_txt2img", device=device)

    loop = pipe.graph.nodes["denoise_loop"]
    inner = loop.graph

    # Новый CFG с меньшим scale (более «свободная» генерация)
    new_guidance = BlockBuilder.build({
        "type": "guidance/cfg",
        "scale": 3.0,
        "guidance_rescale": 0.0,
    })
    inner.replace_node("guidance", new_guidance)
    print("Guidance заменён на CFG с scale=3.0")

    out = pipe("a red apple on a table", num_steps=28, seed=42)
    if out.images:
        out.images[0].save("output_replaced_guidance.png")
        print("Сохранено: output_replaced_guidance.png")


if __name__ == "__main__":
    main()
