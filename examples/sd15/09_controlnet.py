#!/usr/bin/env python3
"""ControlNet для SD 1.5: пространственный контроль (Canny, Depth и др.).

Шаблон controlnet_txt2img добавляет вход control_image. Загружается
ControlNet (по умолчанию Canny). Для работы нужен контрольный силуэт
(например, Canny от реального изображения). Пример генерирует по
промпту и при наличии пути к изображению использует его как референс.

Запуск:
    python examples/sd15/09_controlnet.py
    python examples/sd15/09_controlnet.py path/to/reference.png
"""
import sys
from pathlib import Path

import torch
from PIL import Image
import numpy as np
from yggdrasil.core.graph.graph import ComputeGraph
from yggdrasil.core.graph.executor import GraphExecutor


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    ref_path = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        graph = ComputeGraph.from_template("controlnet_txt2img", device=device)
    except Exception as e:
        print("Шаблон controlnet_txt2img может требовать дополнительных зависимостей:", e)
        print("Используйте sd15_txt2img для базовой генерации.")
        return

    meta = graph.metadata or {}
    channels = meta.get("latent_channels", 4)
    scale = meta.get("spatial_scale_factor", 8)
    h, w = 512 // scale, 512 // scale
    latents = torch.randn(1, channels, h, w, device=graph._device or device, dtype=torch.float32)

    inputs = {
        "prompt": {"text": "a beautiful landscape, detailed"},
        "latents": latents,
    }
    if ref_path and Path(ref_path).exists():
        img = Image.open(ref_path).convert("RGB").resize((512, 512))
        arr = np.array(img).astype(np.float32) / 255.0
        control = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(graph._device or device)
        inputs["control_image"] = control
        print("Используется контрольное изображение:", ref_path)

    executor = GraphExecutor(no_grad=True)
    raw = executor.execute(graph, **inputs)
    decoded = raw.get("decoded")
    if decoded is not None:
        img = decoded.cpu().float()
        if img.min() < -0.01 or img.max() > 1.01:
            img = (img / 2 + 0.5).clamp(0, 1)
        from torchvision.utils import save_image
        save_image(img, "output_controlnet.png")
        print("Сохранено: output_controlnet.png")


if __name__ == "__main__":
    main()
