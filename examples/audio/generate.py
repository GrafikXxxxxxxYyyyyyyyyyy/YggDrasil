#!/usr/bin/env python3
"""Text-to-audio — пример генерации звука по текстовому промпту.

Запуск:
    python examples/audio/generate.py
    python examples/audio/generate.py --template musicldm_txt2audio --prompt "upbeat electronic music"
    python examples/audio/generate.py --template audioldm2_txt2audio --steps 100

Модель скачивается с HuggingFace при первом запуске.
Требуется: CUDA рекомендуется; на CPU генерация может быть очень медленной.
"""
import argparse
import time
import torch
from pathlib import Path

from yggdrasil.pipeline import Pipeline


def save_audio(tensor, path: str, sample_rate: int = 16000) -> None:
    """Сохранить тензор аудио (B, C, T) или (B, T) в WAV."""
    if tensor.dim() == 3:
        tensor = tensor[0]
    if tensor.dim() == 2:
        tensor = tensor[0]
    arr = tensor.float().cpu().numpy()
    # Нормализация в [-1, 1] если нужно
    if arr.max() > 1.0 or arr.min() < -1.0:
        arr = arr / (arr.max() - arr.min() + 1e-8) * 2 - 1
    arr = (arr * 32767).clip(-32768, 32767).astype("int16")
    try:
        import scipy.io.wavfile as wavfile
        wavfile.write(path, sample_rate, arr)
    except ImportError:
        import wave
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(arr.tobytes())


def main():
    parser = argparse.ArgumentParser(description="YggDrasil text-to-audio")
    parser.add_argument("--template", type=str, default="musicldm_txt2audio",
                        help="Шаблон: musicldm_txt2audio, audioldm_txt2audio, audioldm2_txt2audio, stable_audio, dance_diffusion_audio, aura_flow_audio")
    parser.add_argument("--prompt", type=str, default="upbeat electronic music with a catchy melody",
                        help="Текстовый промпт")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="Негативный промпт (для CFG)")
    parser.add_argument("--steps", type=int, default=None, help="Число шагов деноизинга")
    parser.add_argument("--guidance_scale", type=float, default=None, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output_audio.wav", help="Путь к выходному WAV")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate для сохранения WAV (зависит от модели)")
    parser.add_argument("--device", type=str, default=None, help="cuda, mps или cpu")
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Устройство: {device}")

    print(f"Загрузка пайплайна: {args.template}...")
    t0 = time.time()
    pipe = Pipeline.from_template(
        args.template,
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    print(f"Пайплайн готов за {time.time() - t0:.1f}с")

    kwargs = {"prompt": args.prompt, "seed": args.seed}
    if args.negative_prompt:
        kwargs["negative_prompt"] = args.negative_prompt
    if args.steps is not None:
        kwargs["num_steps"] = args.steps
    if args.guidance_scale is not None:
        kwargs["guidance_scale"] = args.guidance_scale

    print(f'Генерация: "{args.prompt}"')
    t0 = time.time()
    output = pipe(**kwargs)
    elapsed = time.time() - t0
    print(f"Генерация завершена за {elapsed:.1f}с")

    if output.audio is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_audio(output.audio, str(out_path), sample_rate=args.sample_rate)
        print(f"Сохранено: {out_path}")
    else:
        print("Аудио не получено. Доступные выходы:", list(output.raw.keys()))


if __name__ == "__main__":
    main()
