#!/usr/bin/env python3
"""Stable Diffusion 1.5 — простой пример генерации изображения.

Запуск:
    python examples/images/sd15/generate.py

Модель скачивается автоматически с HuggingFace при первом запуске.
Требуется ~5 GB свободного места на диске.

Поддерживаемые устройства: CUDA, MPS (Apple Silicon), CPU.
"""
import time
import torch
from yggdrasil.pipeline import Pipeline


def main():
    # ── Определяем устройство ──
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Устройство: {device}")

    # ── Создаём пайплайн из шаблона SD 1.5 ──
    # На CUDA явно задаём float16 (экономия памяти и скорость). Без dtype блоки сохраняют
    # свой dtype из шаблона: UNet/VAE в fp16, CLIP в fp32 (для совместимости с MPS).
    t0 = time.time()
    print("Загрузка модели Stable Diffusion 1.5...")
    pipe = Pipeline.from_template("sd15_txt2img", device=device, dtype=torch.float16)
    print(f"Пайплайн готов за {time.time() - t0:.1f}с: {pipe}")

    # ── Диагностика: проверяем device и dtype ──
    print("\n── Диагностика ──")
    for name, block in pipe.graph._iter_all_blocks():
        params = list(block.parameters())
        if params:
            p = params[0]
            n = sum(p.numel() for p in params)
            mb = n * p.element_size() / 1024 / 1024
            print(f"  {name:30s} {str(p.device):8s} {str(p.dtype):16s} {mb:>7.1f} МБ")
    print()

    # ── Генерация ──
    prompt = "a beautiful mountain landscape at sunset, oil painting, 4k, detailed"

    print(f'Генерация: "{prompt}"')
    t0 = time.time()
    num_steps = 28
    output = pipe(
        prompt,
        guidance_scale=7.5,
        num_steps=num_steps,
        seed=42,
        width=512,
        height=512,
    )
    elapsed = time.time() - t0
    print(f"Генерация завершена за {elapsed:.1f}с ({elapsed/num_steps:.2f}с/шаг)")

    # ── Сохраняем результат ──
    if output.images:
        output.images[0].save("output.png")
        print("Сохранено: output.png")
    else:
        print("Изображение не получено. Доступные выходы:", list(output.raw.keys()))


if __name__ == "__main__":
    main()
