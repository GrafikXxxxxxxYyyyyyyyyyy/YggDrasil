#!/usr/bin/env python3
"""YggDrasil — Stable Diffusion 1.5: минимальный пример.

Самый короткий путь к генерации изображения через Graph API.
Весь бойлерплейт — внутри graph.execute().

    graph = ComputeGraph.from_template("sd15_txt2img", device="cuda")
    outputs = graph.execute(prompt="a cat", num_steps=28, seed=42)
"""
import torch
from yggdrasil.core.graph.graph import ComputeGraph

# ── 1. Устройство (auto-detect) ──
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)
print(f"Устройство: {device}")

# ── 2. Создаём граф и сразу переносим на устройство ──
graph = ComputeGraph.from_template("sd15_txt2img", device=device)
print(f"Граф: {graph}")

# ── 3. Генерируем (всё внутри execute) ──
num_steps = 5 if device in ("mps", "cpu") else 28
print(f"Генерируем ({num_steps} шагов)...")

outputs = graph.execute(
    prompt="a beautiful cyberpunk city at night, highly detailed, 8k",
    guidance_scale=7.5,
    num_steps=num_steps,
    seed=42,
)

# ── 4. Сохраняем ──
image_tensor = outputs["decoded"]
image = (image_tensor / 2 + 0.5).clamp(0, 1)
image = (image[0].cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")

from PIL import Image
from pathlib import Path

output_path = Path("output") / "sd15_hello.png"
output_path.parent.mkdir(exist_ok=True)
Image.fromarray(image).save(output_path)
print(f"Готово! → {output_path}")
