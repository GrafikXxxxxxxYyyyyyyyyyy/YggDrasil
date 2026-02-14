#!/usr/bin/env python3
"""SDXL + ControlNet: подключение ControlNet к графу SDXL одним вызовом.

По умолчанию используется ControlNet по глубине (depth). Для обычных фото
передайте URL или путь — карта глубины может быть получена из изображения
(или используйте готовую depth-карту).

Usage:
    python examples/sdxl/sdxl_with_controlnet.py --control_image "https://example.com/photo.png"
    python examples/sdxl/sdxl_with_controlnet.py --control_image path/to/depth.png
    # Canny вместо depth:
    python examples/sdxl/sdxl_with_controlnet.py --controlnet xinsir/controlnet-sdxl-1.0-canny --control_image canny.png
"""
import argparse

# Качество: 1024x1024, 28–35 steps, guidance_scale 7–7.5. Быстрее: --height 768 --width 768 --steps 20.

# Модели ControlNet SDXL (для --control_type или --controlnet)
CONTROLNET_SDXL_DEPTH = "diffusers/controlnet-depth-sdxl-1.0"
CONTROLNET_SDXL_CANNY = "xinsir/controlnet-sdxl-1.0-canny"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a photo of an astronaut")
    parser.add_argument("--control_image", type=str, default=None, help="Path or URL to control image (depth map or photo for depth ControlNet)")
    parser.add_argument("--controlnet", type=str, default=CONTROLNET_SDXL_DEPTH, help=f"HF model ID (default: {CONTROLNET_SDXL_DEPTH})")
    parser.add_argument("--control_type", type=str, choices=("depth", "canny"), default=None, help="Shortcut: depth -> controlnet-depth-sdxl-1.0, canny -> controlnet-sdxl-1.0-canny")
    parser.add_argument("--output", type=str, default="output_sdxl_controlnet.png")
    parser.add_argument("--steps", type=int, default=28, help="Denoising steps (28–35 for better quality, 20 for speed)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (1024 for quality, 768 for speed)")
    parser.add_argument("--width", type=int, default=1024, help="Image width (1024 for quality, 768 for speed)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG scale (7–7.5 typical for SDXL)")
    parser.add_argument("--conditioning_scale", type=float, default=0.6, help="ControlNet strength (0.5–0.8 for depth)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps"], help="Device (CPU not supported)")
    parser.add_argument("--no_tf32", action="store_true", help="Disable TF32 on CUDA (slower but same precision)")
    args = parser.parse_args()

    if args.device == "cpu":
        import sys
        print("Error: CPU is not supported. Use --device cuda or --device mps.", file=sys.stderr)
        sys.exit(1)

    if args.control_type == "depth":
        args.controlnet = CONTROLNET_SDXL_DEPTH
    elif args.control_type == "canny":
        args.controlnet = CONTROLNET_SDXL_CANNY

    from yggdrasil.pipeline import Pipeline
    from yggdrasil.core.graph import ComputeGraph
    import torch

    if args.device == "cuda" and torch.cuda.is_available() and not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    graph = ComputeGraph.from_template("sdxl_txt2img", device=args.device)
    graph.with_adapter(
        "controlnet",
        controlnet_pretrained=args.controlnet,
        conditioning_scale=args.conditioning_scale,
    )
    pipe = Pipeline(graph=graph, device=args.device)

    kwargs = {
        "prompt": args.prompt,
        "num_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
    }
    if args.control_image:
        kwargs["control_image"] = args.control_image
        print(f"ControlNet: using control image (depth) from {args.control_image!r}, scale={args.conditioning_scale}")
    else:
        dtype = torch.float16 if args.device == "cuda" else torch.float32
        kwargs["control_image"] = torch.zeros(1, 3, args.height, args.width, dtype=dtype, device=args.device)
        print("No --control_image; using zeros (no spatial control from ControlNet)")

    out = pipe(**kwargs)
    if out.images:
        out.images[0].save(args.output)
        print(f"Saved {args.output}")
    else:
        print("No images:", list(out.raw.keys()))


if __name__ == "__main__":
    main()
