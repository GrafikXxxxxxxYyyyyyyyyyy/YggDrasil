#!/usr/bin/env python3
"""Generate an image with Stable Diffusion XL (1024x1024, dual text encoder, added_cond_kwargs).

Usage:
    python examples/images/sdxl/generate_sdxl.py
    python examples/images/sdxl/generate_sdxl.py --prompt "a photo of an astronaut" --height 768 --width 768
"""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse on mars")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default="output_sdxl.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    from yggdrasil.pipeline import Pipeline

    pipe = Pipeline.from_template("sdxl_txt2img", device=args.device)
    out = pipe(
        args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    if out.images:
        out.images[0].save(args.output)
        print(f"Saved {args.output}")
    else:
        print("No images in output:", out.raw.keys())


if __name__ == "__main__":
    main()
