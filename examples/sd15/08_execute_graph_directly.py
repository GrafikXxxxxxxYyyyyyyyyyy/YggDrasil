#!/usr/bin/env python3
"""Прямое выполнение графа без Pipeline: graph.execute(**inputs).

Полный контроль над входами (prompt, negative_prompt, latents, и т.д.)
и доступ к сырому словарю выходов (decoded, latents). Удобно для
интеграции в свои скрипты и для отладки.

Запуск:
    python examples/sd15/08_execute_graph_directly.py
"""
import torch
from yggdrasil.core.graph.graph import ComputeGraph
from yggdrasil.core.graph.executor import GraphExecutor
from yggdrasil.pipeline import Pipeline


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")

    graph = ComputeGraph.from_template("sd15_txt2img", device=device)
    meta = graph.metadata
    channels = meta.get("latent_channels", 4)
    scale = meta.get("spatial_scale_factor", 8)
    h, w = 512 // scale, 512 // scale

    # Генерируем шум вручную (как делает Pipeline)
    g = torch.Generator(device=device if device != "cpu" else "cpu").manual_seed(42)
    latents = torch.randn(1, channels, h, w, device=graph._device or torch.device("cpu"), dtype=torch.float32)

    # Входы графа: prompt (dict с "text"), negative_prompt (str), latents
    inputs = {
        "prompt": {"text": "a mountain at sunset"},
        "negative_prompt": "",
        "latents": latents,
    }

    executor = GraphExecutor(no_grad=True)
    raw = executor.execute(graph, **inputs)

    print("Ключи выхода:", list(raw.keys()))
    decoded = raw.get("decoded")
    if decoded is not None:
        # Приводим к [0, 1] и сохраняем (VAE выдаёт [-1, 1])
        img = decoded.cpu().float()
        if img.min() < -0.01 or img.max() > 1.01:
            img = (img / 2 + 0.5).clamp(0, 1)
        from torchvision.utils import save_image
        save_image(img, "output_direct_execute.png")
        print("Сохранено: output_direct_execute.png")


if __name__ == "__main__":
    main()
