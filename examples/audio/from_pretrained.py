#!/usr/bin/env python3
"""Загрузка аудио-пайплайна по HuggingFace model ID.

Запуск:
    python examples/audio/from_pretrained.py
    python examples/audio/from_pretrained.py --model_id cvssp/audioldm2 --prompt "rain and thunder"
    python examples/audio/from_pretrained.py --model_id ucsd-reach/musicldm
"""
import argparse
import time
import torch
from pathlib import Path

from yggdrasil.pipeline import Pipeline


def save_audio(tensor, path: str, sample_rate: int = 16000) -> None:
    """Сохранить тензор аудио в WAV."""
    if tensor is None:
        return
    if tensor.dim() == 3:
        tensor = tensor[0]
    if tensor.dim() == 2:
        tensor = tensor[0]
    arr = tensor.float().cpu().numpy()
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="ucsd-reach/musicldm",
                        help="HuggingFace model ID (audioldm, audioldm2, musicldm, stabilityai/stable-audio-open-1.0, harmonai/maestro-150k, black-forest-labs/AuraFlow)")
    parser.add_argument("--prompt", type=str, default="upbeat electronic music")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output_audio.wav")
    parser.add_argument("--sample_rate", type=int, default=16000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"Загрузка: {args.model_id}...")
    t0 = time.time()
    pipe = Pipeline.from_pretrained(args.model_id, device=device)
    print(f"Готово за {time.time() - t0:.1f}с")

    kwargs = {"prompt": args.prompt, "seed": args.seed}
    if args.steps is not None:
        kwargs["num_steps"] = args.steps

    print(f'Генерация: "{args.prompt}"')
    output = pipe(**kwargs)
    if output.audio is not None:
        save_audio(output.audio, args.output, sample_rate=args.sample_rate)
        print(f"Сохранено: {args.output}")
    else:
        print("Аудио не получено:", output.raw.keys())


if __name__ == "__main__":
    main()
