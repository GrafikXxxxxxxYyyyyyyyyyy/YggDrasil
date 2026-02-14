#!/usr/bin/env python3
"""Генерация тем же промптом через HuggingFace Diffusers (эталон для сравнения).

Запуск:
    python examples/images/sd15/generate_diffusers.py

Сохраняет output_diffusers.png — сравните с output.png от YggDrasil.
Если diffusers даёт хорошее качество, а YggDrasil нет — баг в нашем пайплайне.
"""
import torch
from diffusers import StableDiffusionPipeline

def main():
    device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Device: {device}")

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        variant='fp16',
        safety_checker=None,
    )
    pipe = pipe.to(device)

    prompt = "a beautiful mountain landscape at sunset, oil painting, 4k, detailed"
    image = pipe(
        prompt,
        guidance_scale=7.5,
        num_inference_steps=28,
        generator=torch.Generator(device=device).manual_seed(42),
    ).images[0]
    image.save("output_diffusers.png")
    print("Saved: output_diffusers.png")

if __name__ == "__main__":
    main()
