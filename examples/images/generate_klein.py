#!/usr/bin/env python3
"""Генерация изображения моделью FLUX.2 [klein] (4B или 9B).

Требуется diffusers из ветки main (0.37.0.dev0+), иначе при загрузке шаблона будет ImportError:
  pip install -r requirements-klein.txt
  # или: pip install "git+https://github.com/huggingface/diffusers.git"
  # или: pip install -e ".[klein]"

Запуск из корня репозитория:
  PYTHONPATH=. python examples/images/generate_klein.py --prompt "Ваш промпт"
  PYTHONPATH=. python examples/images/generate_klein.py --model black-forest-labs/FLUX.2-klein-4B --size 512
"""
import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser(description="Generate image with FLUX.2 Klein")
    p.add_argument("--model", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    p.add_argument("--prompt", type=str, default="A serene landscape with mountains and a lake at sunset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=512, help="Width and height (512, 768 or 1024)")
    p.add_argument("--output", type=str, default="output_klein_4b.png")
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "mps", "cpu"))
    args = p.parse_args()

    import torch
    from yggdrasil.pipeline import Pipeline

    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    elif args.device == "cuda":
        print("CUDA not available, using CPU (will be slow).", file=sys.stderr)
        args.device = "cpu"

    load_kwargs = {"pretrained": args.model, "device": args.device}
    load_kwargs["token"] = os.environ.get("HF_TOKEN") or True

    print(f"Loading {args.model}...")
    pipe = Pipeline.from_template("flux2_klein", **load_kwargs)

    print(f"Generating: {args.prompt!r} ({args.size}x{args.size}, seed={args.seed})")
    out = pipe(
        prompt=args.prompt,
        num_steps=4,
        height=args.size,
        width=args.size,
        guidance_scale=0.0,
        seed=args.seed,
    )

    if not out.images:
        print("No image in output.", file=sys.stderr)
        return 1

    out.images[0].save(args.output)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
