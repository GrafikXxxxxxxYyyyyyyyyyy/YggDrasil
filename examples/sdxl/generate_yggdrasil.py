#!/usr/bin/env python3
"""Генерация изображения через YggDrasil (тот же промпт и seed, что и generate_diffusers.py).

Поддерживает внешние чекпоинты и LoRA с HuggingFace (например https://huggingface.co/OnMoon).
Чекпоинты: OnMoon/sdxl_PhotorealisticMix, OnMoon/sdxl_Juggernaut и др.
LoRA: OnMoon/loras (и другие репозитории в формате diffusers).

Usage:
    python examples/sdxl/generate_yggdrasil.py --prompt "A cat on the floor" --seed 42
    python examples/sdxl/generate_yggdrasil.py --checkpoint OnMoon/sdxl_PhotorealisticMix
    python examples/sdxl/generate_yggdrasil.py --lora OnMoon/loras --lora_weight SomeLora.safetensors
"""
import argparse
import sys

# Те же константы, что и в generate_diffusers.py — для честного сравнения
DEFAULT_PROMPT = "A photorealistic cat sitting on a wooden floor, soft lighting"
DEFAULT_SEED = 42
DEFAULT_STEPS = 28
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_GUIDANCE = 7.5


def main():
    p = argparse.ArgumentParser(description="Generate image with YggDrasil (SDXL)")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (same as diffusers script)")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--guidance_scale", type=float, default=DEFAULT_GUIDANCE)
    p.add_argument(
        "--checkpoint",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base checkpoint: HF id (e.g. OnMoon/sdxl_PhotorealisticMix) or path",
    )
    p.add_argument("--lora", type=str, default=None, help="LoRA repo (e.g. OnMoon/loras)")
    p.add_argument("--lora_weight", type=str, default=None, help="LoRA file name in repo (.safetensors)")
    p.add_argument("--output", type=str, default="output_yggdrasil.png")
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "mps", "cpu"))
    args = p.parse_args()

    import torch
    from yggdrasil.pipeline import Pipeline

    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Loading checkpoint: {args.checkpoint}")
    pipe = Pipeline.from_template(
        "sdxl_txt2img",
        pretrained=args.checkpoint,
        device=args.device,
    )

    if args.lora:
        print(f"Loading LoRA: {args.lora} (weight={args.lora_weight or 'auto'})")
        pipe.load_lora_weights(
            args.lora,
            weight_name=args.lora_weight or None,
        )

    print(f"Generating: prompt={args.prompt!r}, seed={args.seed}, steps={args.steps}")
    out = pipe(
        prompt=args.prompt,
        num_steps=args.steps,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    if out.images:
        out.images[0].save(args.output)
        print(f"Saved {args.output}")
    else:
        print("No images in output:", list(out.raw.keys()), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
