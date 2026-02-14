#!/usr/bin/env python3
"""Кастомные параметры генерации: guidance_scale, num_steps, negative_prompt.

Показывает, что все параметры передаются в pipe() и попадают в graph.execute()
без пересборки графа. Меняем силу текста, число шагов и негативный промпт.

Запуск:
    python examples/images/sd15/04_custom_parameters.py
"""
import torch
from yggdrasil.pipeline import Pipeline


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    pipe = Pipeline.from_template("sd15_txt2img", device=device)

    prompt = "a cute robot, digital art"
    out = pipe(
        prompt,
        guidance_scale=10.0,   # сильнее следование промпту
        num_steps=30,
        negative_prompt="blurry, low quality, distorted",
        seed=999,
        width=512,
        height=512,
    )

    if out.images:
        out.images[0].save("output_custom_params.png")
        print("Сохранено: output_custom_params.png (guidance_scale=10, num_steps=30, negative_prompt задан)")


if __name__ == "__main__":
    main()
