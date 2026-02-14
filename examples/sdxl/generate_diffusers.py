#!/usr/bin/env python3
"""Generate image with HuggingFace Diffusers (reference for comparison with YggDrasil).

Same prompt and seed as generate_yggdrasil.py. Supports external checkpoints and LoRA
from HuggingFace (e.g. https://huggingface.co/OnMoon).

Usage:
    python examples/sdxl/generate_diffusers.py --prompt "A cat on the floor" --seed 42
    python examples/sdxl/generate_diffusers.py --checkpoint OnMoon/sdxl_PhotorealisticMix
    python examples/sdxl/generate_diffusers.py --lora OnMoon/loras --lora_weight SomeLora.safetensors
"""
import argparse
import sys

DEFAULT_PROMPT = "A photorealistic cat sitting on a wooden floor, soft lighting"
DEFAULT_SEED = 42
DEFAULT_STEPS = 28
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_GUIDANCE = 7.5


def main():
    p = argparse.ArgumentParser(description="Generate image with diffusers (SDXL)")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--guidance_scale", type=float, default=DEFAULT_GUIDANCE)
    p.add_argument("--checkpoint", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--lora", type=str, default=None)
    p.add_argument("--lora_weight", type=str, default=None)
    p.add_argument("--lora_scale", type=float, default=0.8)
    p.add_argument("--output", type=str, default="output_diffusers.png")
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "mps", "cpu"))
    args = p.parse_args()

    import torch
    from diffusers import StableDiffusionXLPipeline

    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    print("Loading checkpoint:", args.checkpoint)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.checkpoint, torch_dtype=dtype, use_safetensors=True
    )
    pipe = pipe.to(args.device)

    if args.lora:
        print("Loading LoRA:", args.lora)
        pipe.load_lora_weights(args.lora, weight_name=args.lora_weight or None)
        pipe.set_adapters(["default"], adapter_weights=[args.lora_scale])

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    print("Generating: prompt=%r seed=%s steps=%s" % (args.prompt, args.seed, args.steps))
    out = pipe(
        args.prompt,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    out.images[0].save(args.output)
    print("Saved", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
