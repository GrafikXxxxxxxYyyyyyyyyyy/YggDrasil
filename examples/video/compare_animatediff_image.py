#!/usr/bin/env python3
"""Сравнение анимации изображения: Diffusers AnimateDiff vs YggDrasil AnimateDiff.

Режим image-to-video: исходное изображение — первый кадр, оба пайплайна генерируют
видео, сохраняющее связь с этим изображением (кодируем картинку в латенты, добавляем
шум, деноизим по prompt).

- Diffusers: AnimateDiffVideoToVideoPipeline(video=[image]*N, strength=...).
- YggDrasil: кодируем изображение VAE, повторяем по кадрам, добавляем шум по расписанию,
  передаём latents в пайплайн и деноизим.

Запуск:
    python examples/video/compare_animatediff_image.py
    python examples/video/compare_animatediff_image.py --image /path/to/img.png --prompt "gentle motion" --strength 0.7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = PROJECT_ROOT / "output_yggdrasil_fixed.png"
OUTPUT_DIFFUSERS = PROJECT_ROOT / "examples" / "video" / "video_animatediff_diffusers.mp4"
OUTPUT_YGGDRASIL = PROJECT_ROOT / "examples" / "video" / "video_animatediff_yggdrasil.mp4"
FPS = 8


def get_image_size(path: Path) -> tuple[int, int]:
    """Вернуть (width, height), выровненные до кратного 8."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
    except Exception:
        w, h = 512, 512
    w = (w // 8) * 8
    h = (h // 8) * 8
    return min(max(w, 256), 768), min(max(h, 256), 768)


def load_image_as_tensor(path: Path, width: int, height: int, device: torch.device) -> torch.Tensor:
    """Загрузить изображение как тензор (1, 3, H, W) в [-1, 1] для VAE."""
    from PIL import Image
    import torch.nn.functional as F
    pil = Image.open(path).convert("RGB")
    arr = torch.from_numpy(__import__("numpy").array(pil)).float() / 255.0
    t = arr.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    if t.shape[2] != height or t.shape[3] != width:
        t = F.interpolate(t, size=(height, width), mode="bilinear", align_corners=False)
    t = t * 2.0 - 1.0  # [0,1] -> [-1,1] для SD VAE
    return t.to(device=device)


def save_video_tensor(tensor: torch.Tensor, path: Path, fps: float = FPS) -> bool:
    """Сохранить тензор (1, C, T, H, W) в [0,1] или [-1,1] как MP4 или кадры."""
    if tensor is None or tensor.dim() != 5:
        return False
    t = tensor[0].float().cpu()
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1, 1)
    if t.dim() == 4 and t.shape[0] == 3:
        t = t.permute(1, 2, 3, 0)  # (T,H,W,C)
    if t.min() < -0.01 or t.max() > 1.01:
        t = (t / 2 + 0.5).clamp(0, 1)
    t = (t.numpy() * 255).clip(0, 255).astype("uint8")
    try:
        import imageio
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(path), list(t), fps=fps, codec="libx264")
        return True
    except (ImportError, ValueError):
        pass
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
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        for i in range(t.shape[0]):
            Image.fromarray(t[i]).save(path.parent / f"{path.stem}_frame_{i:04d}.png")
        print(f"MP4 недоступен (установите imageio[ffmpeg]). Кадры: {path.parent / (path.stem + '_frame_*.png')}", file=sys.stderr)
        return False
    except Exception:
        return False


def run_diffusers(
    image_path: Path,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    num_steps: int,
    strength: float,
    seed: int,
    device: str,
    output_path: Path,
) -> bool:
    """Image-to-video через AnimateDiffVideoToVideoPipeline: видео = N копий изображения + strength."""
    try:
        from diffusers import AnimateDiffVideoToVideoPipeline, DDIMScheduler, MotionAdapter
        from PIL import Image
    except ImportError as e:
        print(f"Diffusers AnimateDiff недоступен: {e}", file=sys.stderr)
        return False

    print("Загрузка Diffusers AnimateDiff (VideoToVideo)...")
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
    pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        motion_adapter=adapter,
        torch_dtype=torch.float32,
    )
    pipe.scheduler = DDIMScheduler(
        beta_schedule="linear",
        steps_offset=1,
        clip_sample=False,
        timestep_spacing="linspace",
    )
    pipe = pipe.to(device)

    pil_img = Image.open(image_path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    video_frames = [pil_img] * num_frames  # первый кадр = изображение, остальные копии для консистентности

    generator = torch.Generator(device=device).manual_seed(seed) if seed >= 0 else None
    print(f"Генерация (diffusers img2vid): {num_frames} кадров, {width}x{height}, strength={strength}...")
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or "",
        video=video_frames,
        strength=strength,
        num_inference_steps=num_steps,
        height=height,
        width=width,
        generator=generator,
    )
    if not out.frames or not out.frames[0]:
        print("Diffusers не вернул кадры.", file=sys.stderr)
        return False
    pil_list = out.frames[0]
    import numpy as np
    arr = np.stack([np.array(p) for p in pil_list], axis=0).astype(np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr[:, :, :, np.newaxis].repeat(3, axis=-1)
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)
    ok = save_video_tensor(tensor, output_path, fps=FPS)
    if ok:
        print(f"Сохранено (diffusers): {output_path}")
    return ok


def run_yggdrasil(
    image_path: Path,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    num_steps: int,
    strength: float,
    seed: int,
    device: str,
    output_path: Path,
) -> bool:
    """Image-to-video через YggDrasil: кодируем изображение в латенты, добавляем шум, деноизим."""
    try:
        from yggdrasil.pipeline import Pipeline
    except ImportError as e:
        print(f"YggDrasil недоступен: {e}", file=sys.stderr)
        return False

    print("Загрузка YggDrasil AnimateDiff...")
    pipe = Pipeline.from_template("animatediff_txt2vid", device=device)
    graph = pipe.graph
    dev = graph._device or torch.device("cpu")

    codec = graph.nodes.get("codec")
    loop = graph.nodes.get("denoise_loop")
    if not codec or not hasattr(codec, "encode") or not loop or not hasattr(loop, "graph"):
        print("Граф без codec или denoise_loop.", file=sys.stderr)
        return False

    inner = loop.graph
    solver = inner.nodes.get("solver") if inner else None
    if not solver or not hasattr(solver, "alphas_cumprod"):
        print("Солвер без alphas_cumprod (нужен DDIM-совместимый).", file=sys.stderr)
        return False

    # Таймстепы как в цикле (leading); strength: с какого шага начинать (0.7 = меньше шума, ближе к картинке)
    T = int(getattr(loop, "num_train_timesteps", 1000))
    step_ratio = T // num_steps
    timesteps_full = torch.arange(0, num_steps, device=dev).long() * step_ratio
    timesteps_full = timesteps_full.flip(0)
    steps_offset = int(getattr(loop, "steps_offset", 0))
    timesteps_full = (timesteps_full + steps_offset).clamp(0, T - 1)
    start_idx = min(int((1.0 - strength) * num_steps), num_steps - 1)
    start_idx = max(0, start_idx)
    timesteps = timesteps_full[start_idx:]
    first_t = timesteps[0].item()

    # Изображение -> латент, повторить по кадрам
    pixel = load_image_as_tensor(image_path, width, height, dev)
    with torch.no_grad():
        image_latent = codec.encode(pixel)  # (1, C, h, w)
    scale = graph.metadata.get("spatial_scale_factor", 8)
    h, w = height // scale, width // scale
    video_latent = image_latent.unsqueeze(2).expand(1, image_latent.shape[1], num_frames, h, w).clone()

    # Добавить шум на шаге first_t (как в img2img)
    alpha = solver.alphas_cumprod[first_t].to(device=dev, dtype=video_latent.dtype)
    while alpha.dim() < video_latent.dim():
        alpha = alpha.unsqueeze(-1)
    g = torch.Generator(device=dev).manual_seed(seed) if seed >= 0 else None
    noise = torch.randn_like(video_latent, device=dev, generator=g)
    noisy_latent = alpha.sqrt() * video_latent + (1 - alpha).sqrt() * noise

    actual_seed = int(seed) if seed >= 0 else torch.randint(0, 2**32, (1,), device=dev).item()
    print(f"Генерация (YggDrasil img2vid): {num_frames} кадров, {width}x{height}, strength={strength}, шагов={len(timesteps)}...")
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or "",
        width=width,
        height=height,
        num_frames=num_frames,
        num_steps=len(timesteps),
        seed=actual_seed,
        latents=noisy_latent,
        timesteps=timesteps,
    )
    if out.video is None:
        print("YggDrasil не вернул видео. Keys:", getattr(out, "raw", {}).keys(), file=sys.stderr)
        return False
    ok = save_video_tensor(out.video, output_path, fps=FPS)
    if ok:
        print(f"Сохранено (YggDrasil): {output_path}")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Сравнение image-to-video: Diffusers vs YggDrasil AnimateDiff по одному изображению."
    )
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE), help="Путь к изображению для анимации")
    parser.add_argument("--prompt", type=str, default="gentle motion, high quality, subtle animation")
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--strength", type=float, default=0.7, help="Сила изменения (0.5–1.0). Выше = больше движения от исходного кадра.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-diffusers", type=str, default=str(OUTPUT_DIFFUSERS))
    parser.add_argument("--out-yggdrasil", type=str, default=str(OUTPUT_YGGDRASIL))
    parser.add_argument("--skip-diffusers", action="store_true", help="Не запускать diffusers")
    parser.add_argument("--skip-yggdrasil", action="store_true", help="Не запускать YggDrasil")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"Изображение не найдено: {image_path}", file=sys.stderr)
        sys.exit(1)

    width, height = get_image_size(image_path)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Изображение: {image_path.name}, разрешение: {width}x{height}, device={device}, strength={args.strength}")

    out_diffusers = Path(args.out_diffusers)
    out_yggdrasil = Path(args.out_yggdrasil)

    ok_diffusers = True
    ok_yggdrasil = True

    if not args.skip_diffusers:
        ok_diffusers = run_diffusers(
            image_path=image_path,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=width,
            height=height,
            num_frames=args.num_frames,
            num_steps=args.steps,
            strength=args.strength,
            seed=args.seed,
            device=device,
            output_path=out_diffusers,
        )
    else:
        print("Пропуск Diffusers (--skip-diffusers).")

    if not args.skip_yggdrasil:
        ok_yggdrasil = run_yggdrasil(
            image_path=image_path,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=width,
            height=height,
            num_frames=args.num_frames,
            num_steps=args.steps,
            strength=args.strength,
            seed=args.seed,
            device=device,
            output_path=out_yggdrasil,
        )
    else:
        print("Пропуск YggDrasil (--skip-yggdrasil).")

    print("\n--- Итог ---")
    if ok_diffusers:
        print(f"  Diffusers:  {out_diffusers}")
    else:
        print("  Diffusers:  ошибка или пропущен")
    if ok_yggdrasil:
        print(f"  YggDrasil:  {out_yggdrasil}")
    else:
        print("  YggDrasil:  ошибка или пропущен")

    sys.exit(0 if (ok_diffusers or ok_yggdrasil) else 1)


if __name__ == "__main__":
    main()
