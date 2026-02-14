#!/usr/bin/env python3
"""Тест AnimateDiff по референсному изображению: размеры берутся из output_yggdrasil_fixed.png, результат в корень проекта.

Запуск:
    python examples/video/test_animatediff_image.py
    python examples/video/test_animatediff_image.py --prompt "a cat walking" --steps 20 --num_frames 8
"""
import argparse
import sys
from pathlib import Path

import torch

# корень репозитория
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REF_IMAGE = PROJECT_ROOT / "output_yggdrasil_fixed.png"
OUTPUT_VIDEO = PROJECT_ROOT / "animatediff_output.mp4"
OUTPUT_FALLBACK = PROJECT_ROOT / "animatediff_frames"


def get_image_size(path: Path):
    """Вернуть (width, height) из изображения."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        return 512, 512


def video_tensor_to_mp4(tensor: torch.Tensor, path: Path, fps: int = 8) -> bool:
    """Сохранить тензор (1, C, T, H, W) в [0,1] или [-1,1] как MP4."""
    if tensor is None or tensor.dim() != 5:
        return False
    t = tensor[0].float().cpu()
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1, 1)
    if t.dim() == 4 and t.shape[0] == 3:
        t = t.permute(1, 2, 3, 0)
    if t.min() < -0.01 or t.max() > 1.01:
        t = (t / 2 + 0.5).clamp(0, 1)
    t = (t.numpy() * 255).clip(0, 255).astype("uint8")
    try:
        import imageio
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(path), list(t), fps=fps, codec="libx264")
        return True
    except ImportError:
        pass
    except ValueError as e:
        if "backend" in str(e).lower() or "ffmpeg" in str(e).lower():
            print("Для сохранения MP4 установите: pip install imageio[ffmpeg]", file=__import__("sys").stderr)
    try:
        import cv2
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        H, W = t.shape[1], t.shape[2]
        out = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
        for i in range(t.shape[0]):
            out.write(cv2.cvtColor(t[i], cv2.COLOR_RGB2BGR))
        out.release()
        return True
    except ImportError:
        pass
    # Fallback: сохранить кадры как PNG
    OUTPUT_FALLBACK.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        for i in range(t.shape[0]):
            Image.fromarray(t[i]).save(OUTPUT_FALLBACK / f"frame_{i:04d}.png")
        print(f"Video saved as frames in {OUTPUT_FALLBACK}")
        return False
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="AnimateDiff test from reference image dimensions")
    parser.add_argument("--image", type=str, default=str(REF_IMAGE), help="Reference image for size")
    parser.add_argument("--prompt", type=str, default="a beautiful landscape, gentle motion, high quality")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(OUTPUT_VIDEO))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    ref_path = Path(args.image)
    if not ref_path.is_file():
        print(f"Reference image not found: {ref_path}", file=sys.stderr)
        print("Using 512x512.", file=sys.stderr)
        width, height = 512, 512
    else:
        width, height = get_image_size(ref_path)
        # AnimateDiff/SD 1.5 часто 512; выравниваем до кратного 8
        width = (width // 8) * 8
        height = (height // 8) * 8
        width = min(max(width, 256), 768)
        height = min(max(height, 256), 768)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading AnimateDiff (animatediff_txt2vid)...")
    from yggdrasil.pipeline import Pipeline

    pipe = Pipeline.from_template("animatediff_txt2vid", device=device)
    print(f"Generating: {args.num_frames} frames, {width}x{height}, steps={args.steps}, prompt={args.prompt!r}")
    out = pipe(
        prompt=args.prompt,
        width=width,
        height=height,
        num_steps=args.steps,
        num_frames=args.num_frames,
        seed=args.seed,
    )
    if out.video is None:
        print("No video in output. Keys:", getattr(out, "raw", {}).keys(), file=sys.stderr)
        return 1
    out_path = Path(args.output)
    if video_tensor_to_mp4(out.video, out_path):
        print(f"Saved: {out_path}")
    else:
        print(f"Frames saved to {OUTPUT_FALLBACK}; install imageio or opencv-python for MP4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
