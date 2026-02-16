#!/usr/bin/env python3
"""
Compare Diffusers vs YggDrasil for image models: SD 1.5, SDXL, SD3, FLUX.

Same prompt, seed, steps, guidance_scale -> diffusers.png vs yggdrasil.png.
Optional: --compare computes MSE/PSNR when both images exist.

Run from repo root:
  python examples/images/compare_diffusers_yggdrasil.py --model sd15
  python examples/images/compare_diffusers_yggdrasil.py --model sdxl --yggdrasil-only
  python examples/images/compare_diffusers_yggdrasil.py --model sd3 --prompt "a cat" --steps 28
  python examples/images/compare_diffusers_yggdrasil.py --model flux --steps 28
  python examples/images/compare_diffusers_yggdrasil.py --model sd15 --compare  # metric after both generated
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_MODELS = {
    "sd15": {
        "diffusers_pipeline": "StableDiffusionPipeline",
        "diffusers_model": "runwayml/stable-diffusion-v1-5",
        "yggdrasil_template": "sd15_txt2img",
        "width": 512,
        "height": 512,
        "steps": 28,
        "guidance_scale": 7.5,
        "negative_prompt": None,
    },
    "sdxl": {
        "diffusers_pipeline": "StableDiffusionXLPipeline",
        "diffusers_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "yggdrasil_template": "sdxl_txt2img",
        "width": 1024,
        "height": 1024,
        "steps": 30,
        "guidance_scale": 7.5,
        "negative_prompt": "",
    },
    "sd3": {
        "diffusers_pipeline": "StableDiffusion3Pipeline",
        "diffusers_model": "stabilityai/stable-diffusion-3-medium-diffusers",
        "yggdrasil_template": "sd3_txt2img",
        "width": 1024,
        "height": 1024,
        "steps": 28,
        "guidance_scale": 5.0,
        "negative_prompt": "",
    },
    "flux": {
        "diffusers_pipeline": "FluxPipeline",
        "diffusers_model": "black-forest-labs/FLUX.1-dev",
        "yggdrasil_template": "flux_txt2img",
        "width": 1024,
        "height": 1024,
        "steps": 28,
        "guidance_scale": 3.5,
        "negative_prompt": "",
    },
}


def _check_diffusers_deps(model_key: str) -> None:
    """Ensure torch and required diffusers pipeline are available. Raises SystemExit on failure."""
    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit(
            "torch is required for Diffusers comparison. Install: pip install torch diffusers"
        ) from None
    pipe_name = _MODELS.get(model_key, {}).get("diffusers_pipeline", "")
    if not pipe_name:
        return
    pipe_map = {}
    try:
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusion3Pipeline
        pipe_map.update({
            "StableDiffusionPipeline": StableDiffusionPipeline,
            "StableDiffusionXLPipeline": StableDiffusionXLPipeline,
            "StableDiffusion3Pipeline": StableDiffusion3Pipeline,
        })
    except ImportError as e:
        raise SystemExit(
            "diffusers not installed or missing pipelines. Install: pip install diffusers. Error: %s" % e
        ) from None
    if pipe_name == "FluxPipeline":
        try:
            from diffusers import FluxPipeline
            pipe_map["FluxPipeline"] = FluxPipeline
        except ImportError as e:
            raise SystemExit(
                "FluxPipeline requires diffusers>=0.27. Install: pip install 'diffusers>=0.27'. Error: %s" % e
            ) from None
    if pipe_name not in pipe_map or pipe_map[pipe_name] is None:
        raise SystemExit("Diffusers pipeline %r not available for model %s" % (pipe_name, model_key))


def _get_device(device=None):
    if device:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        )
    except ImportError:
        return "cpu"


def _free_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
    except Exception:
        pass


def _run_diffusers(cfg, prompt, num_steps, guidance_scale, width, height, seed, device):
    import torch
    pipe_map = {}
    try:
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusion3Pipeline
        pipe_map.update({
            "StableDiffusionPipeline": StableDiffusionPipeline,
            "StableDiffusionXLPipeline": StableDiffusionXLPipeline,
            "StableDiffusion3Pipeline": StableDiffusion3Pipeline,
        })
    except ImportError:
        pass
    try:
        from diffusers import FluxPipeline
        pipe_map["FluxPipeline"] = FluxPipeline
    except ImportError:
        pass
    pipe_cls = pipe_map.get(cfg["diffusers_pipeline"])
    if pipe_cls is None:
        raise ImportError(
            "Diffusers pipeline %r not available. Install: pip install diffusers; "
            "for FLUX ensure diffusers>=0.27" % cfg["diffusers_pipeline"]
        )
    dtype = torch.float16 if device == "cuda" else torch.float32
    print("Loading Diffusers pipeline (%s)..." % cfg["diffusers_model"])
    pipe = pipe_cls.from_pretrained(cfg["diffusers_model"], torch_dtype=dtype)
    pipe = pipe.to(device)
    if device == "cpu":
        pipe = pipe.to(torch.float32)
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    kwargs = dict(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )
    if cfg.get("negative_prompt") is not None:
        kwargs["negative_prompt"] = cfg["negative_prompt"]
    # SDXL: explicit guidance_rescale=0.0 for parity (Diffusers default)
    if cfg.get("diffusers_pipeline") == "StableDiffusionXLPipeline":
        kwargs.setdefault("guidance_rescale", 0.0)
    print("Diffusers: generating (steps=%s, guidance_scale=%s, seed=%s)..." % (num_steps, guidance_scale, seed))
    result = pipe(**kwargs)
    img = result.images[0]
    del result, pipe
    return img


def _run_yggdrasil(cfg, prompt, num_steps, guidance_scale, width, height, seed, device):
    from yggdrasil.pipeline import InferencePipeline
    template = cfg["yggdrasil_template"]
    print("Loading YggDrasil pipeline (%s)..." % template)
    pipe = InferencePipeline.from_template(template, device=device)
    print("YggDrasil: generating (steps=%s, guidance_scale=%s, seed=%s)..." % (num_steps, guidance_scale, seed))
    kwargs = dict(
        prompt=prompt,
        num_steps=num_steps,
        seed=seed,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
    )
    if cfg.get("negative_prompt") is not None:
        kwargs["negative_prompt"] = cfg["negative_prompt"]
    output = pipe(**kwargs)
    if output.images:
        return output.images[0]
    raise RuntimeError("YggDrasil returned no images. Raw keys: %s" % list(output.raw.keys()))


def _compare_images(path_a: Path, path_b: Path) -> dict:
    """Compare two images: MSE, PSNR, max diff. Returns dict or empty if error."""
    try:
        import numpy as np
        from PIL import Image
        a = np.array(Image.open(path_a).convert("RGB"), dtype=np.float32) / 255.0
        b = np.array(Image.open(path_b).convert("RGB"), dtype=np.float32) / 255.0
        if a.shape != b.shape:
            return {"error": "shape mismatch: %s vs %s" % (a.shape, b.shape)}
        mse = float(np.mean((a - b) ** 2))
        if mse < 1e-10:
            psnr = 100.0
        else:
            psnr = 10.0 * np.log10(1.0 / mse)
        max_diff = float(np.max(np.abs(a - b)))
        return {"mse": mse, "psnr_db": psnr, "max_diff": max_diff}
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Compare Diffusers vs YggDrasil (same params)")
    parser.add_argument("--model", choices=list(_MODELS), default="sd15", help="sd15, sdxl, sd3, flux")
    parser.add_argument("--prompt", type=str, default="a red vintage bicycle leaning against a yellow wall, sunny day, photorealistic")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--diffusers-only", action="store_true")
    parser.add_argument("--yggdrasil-only", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Compute MSE/PSNR when both images exist")
    args = parser.parse_args()

    cfg = _MODELS[args.model].copy()
    for k in ("steps", "guidance_scale", "width", "height"):
        val = getattr(args, k, None)
        if val is not None:
            cfg[k] = val
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / args.model / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _get_device(args.device)

    print("Settings: model=%s prompt=%r steps=%s guidance_scale=%s seed=%s size=%sx%s device=%s" % (
        args.model, args.prompt, cfg["steps"], cfg["guidance_scale"], args.seed,
        cfg["width"], cfg["height"], device,
    ))

    if not args.yggdrasil_only:
        _check_diffusers_deps(args.model)

    if not args.yggdrasil_only:
        img = _run_diffusers(cfg, args.prompt, cfg["steps"], cfg["guidance_scale"],
                             cfg["width"], cfg["height"], args.seed, device)
        img.save(out_dir / "diffusers.png")
        print("Saved: %s" % (out_dir / "diffusers.png"))
        _free_gpu()

    if not args.diffusers_only:
        img = _run_yggdrasil(cfg, args.prompt, cfg["steps"], cfg["guidance_scale"],
                             cfg["width"], cfg["height"], args.seed, device)
        img.save(out_dir / "yggdrasil.png")
        print("Saved: %s" % (out_dir / "yggdrasil.png"))

    diff_path = out_dir / "diffusers.png"
    ygg_path = out_dir / "yggdrasil.png"
    print("Done. Compare: %s (Diffusers) vs %s (YggDrasil)" % (diff_path, ygg_path))

    if args.compare and diff_path.exists() and ygg_path.exists():
        metrics = _compare_images(diff_path, ygg_path)
        if "error" in metrics:
            print("Compare error: %s" % metrics["error"])
        else:
            print("Metrics: MSE=%.6f PSNR=%.2f dB max_diff=%.4f" % (
                metrics["mse"], metrics["psnr_db"], metrics["max_diff"],
            ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
