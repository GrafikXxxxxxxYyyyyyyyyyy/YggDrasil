#!/usr/bin/env python3
"""Загрузка SD 1.5 по имени пресета или HuggingFace model ID.

Демонстрирует: Pipeline.from_pretrained() — единая точка входа для
пресетов ("sd15") и HF ID ("runwayml/stable-diffusion-v1-5").

Запуск:
    python examples/sd15/02_from_pretrained.py
"""
import torch
from yggdrasil.pipeline import Pipeline


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}\n")

    # По имени пресета ("sd15" -> sd15_txt2img) или по HuggingFace model ID
    model_id = "sd15"  # или "runwayml/stable-diffusion-v1-5"
    print(f"Загрузка: {model_id}...")
    pipe = Pipeline.from_pretrained(model_id, device=device)
    out = pipe("a red apple on a table", num_steps=20, seed=123)
    if out.images:
        out.images[0].save("output_from_pretrained.png")
        print("Сохранено: output_from_pretrained.png")


if __name__ == "__main__":
    main()
