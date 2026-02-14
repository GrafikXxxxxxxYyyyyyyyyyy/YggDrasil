#!/usr/bin/env python3
"""Image-to-image: шаблон sd15_img2img и вход source_image.

Шаблон sd15_img2img добавляет к графу вход source_image (подключён к codec).
Пример показывает загрузку шаблона и вызов pipe() с опциональным путём к изображению.
Полный img2img (старт из encoded image + noise) требует передачи латентов вручную.

Запуск:
    python examples/images/sd15/03_img2img.py
    python examples/images/sd15/03_img2img.py path/to/image.png
"""
import sys
from pathlib import Path

import torch
from PIL import Image
from yggdrasil.pipeline import Pipeline


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    pipe = Pipeline.from_template("sd15_img2img", device=device)

    if image_path and Path(image_path).exists():
        img = Image.open(image_path).convert("RGB").resize((512, 512))
        import numpy as np
        arr = np.array(img).astype(np.float32) / 255.0
        source = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        out = pipe(
            "same scene, oil painting style",
            source_image=source,
            num_steps=28,
            seed=42,
        )
    else:
        out = pipe("a landscape, oil painting", num_steps=28, seed=42)

    if out.images:
        out.images[0].save("output_img2img.png")
        print("Сохранено: output_img2img.png")
    else:
        print("Нет изображения в выводе. Доступные ключи:", list(getattr(out, "raw", out).keys()))


if __name__ == "__main__":
    main()
