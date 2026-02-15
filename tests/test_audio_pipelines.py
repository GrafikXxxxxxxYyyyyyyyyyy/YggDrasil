#!/usr/bin/env python3
"""Тест всех аудио-пайплайнов: генерация по одному промпту, сохранение WAV и сводка.

Запуск:
    python tests/test_audio_pipelines.py
    python tests/test_audio_pipelines.py --steps 15 --output-dir /tmp/audio_test

При нехватке места на диске очищается кэш HuggingFace (HF_HOME / ~/.cache/huggingface).
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import torch

# минимум шагов для быстрой проверки
DEFAULT_STEPS = 20
TEST_PROMPT = "soft piano melody"
OUTPUT_DIR_DEFAULT = "tests/audio_outputs"


def clear_huggingface_cache():
    """Очистить кэш HuggingFace для освобождения места."""
    cache_dir = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    if not os.path.isdir(cache_dir):
        return
    try:
        shutil.rmtree(cache_dir)
        print(f"[CACHE] Очищен кэш HuggingFace: {cache_dir}")
    except Exception as e:
        print(f"[CACHE] Не удалось очистить кэш: {e}", file=sys.stderr)


def save_audio(tensor, path: str, sample_rate: int = 16000) -> None:
    if tensor is None:
        return
    if tensor.dim() == 3:
        tensor = tensor[0]
    if tensor.dim() == 2:
        tensor = tensor[0]
    arr = tensor.float().cpu().numpy()
    if arr.size == 0:
        return
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


def run_one(template: str, prompt: str, steps: int, output_dir: Path, device: str, dtype) -> dict:
    """Запуск одного пайплайна. Возвращает dict с результатом."""
    from yggdrasil.pipeline import InferencePipeline

    out_path = output_dir / f"{template}.wav"
    result = {
        "template": template,
        "success": False,
        "path": str(out_path),
        "load_time_s": None,
        "gen_time_s": None,
        "error": None,
        "audio_shape": None,
    }
    try:
        t0 = time.time()
        pipe = InferencePipeline.from_template(template, device=device, dtype=dtype)
        result["load_time_s"] = round(time.time() - t0, 1)
        kwargs = {"prompt": prompt, "seed": 42, "num_steps": steps}
        t0 = time.time()
        output = pipe(**kwargs)
        result["gen_time_s"] = round(time.time() - t0, 1)
        if output.audio is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_audio(output.audio, str(out_path), sample_rate=16000)
            result["success"] = True
            result["audio_shape"] = list(output.audio.shape)
        else:
            result["error"] = "output.audio is None; keys: " + str(list(getattr(output, "raw", {}).keys()))
    except Exception as e:
        result["error"] = str(e)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Число шагов деноизинга")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT, help="Папка для WAV")
    parser.add_argument("--prompt", type=str, default=TEST_PROMPT)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip", type=str, nargs="*", default=[], help="Шаблоны пропустить (например stable_audio)")
    parser.add_argument("--include-all", action="store_true", help="Не пропускать по умолчанию dance_diffusion_audio и aura_flow_audio")
    parser.add_argument("--clear-cache", action="store_true", help="Перед тестами очистить кэш HuggingFace")
    args = parser.parse_args()

    if args.clear_cache:
        clear_huggingface_cache()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    output_dir = Path(args.output_dir)

    from yggdrasil.core.graph.templates.audio_pipelines import AUDIO_TEMPLATES

    skip = set(args.skip)
    if not args.include_all:
        skip |= {"dance_diffusion_audio", "aura_flow_audio"}
    templates = [t for t in AUDIO_TEMPLATES if t not in skip]
    print(f"Устройство: {device}, шаги: {args.steps}, промпт: {args.prompt!r}")
    print(f"Шаблоны: {templates}")
    print(f"Выход: {output_dir.absolute()}\n")

    results = []
    for i, template in enumerate(templates):
        print(f"[{i+1}/{len(templates)}] {template}...", end=" ", flush=True)
        try:
            res = run_one(template, args.prompt, args.steps, output_dir, device, dtype)
        except Exception as e:
            res = {
                "template": template,
                "success": False,
                "path": "",
                "load_time_s": None,
                "gen_time_s": None,
                "error": str(e),
                "audio_shape": None,
            }
        results.append(res)
        if res["success"]:
            print(f"OK {res['gen_time_s']}s -> {res['path']}")
        else:
            print(f"FAIL: {res.get('error', 'no audio')}")
        err_lower = str(res.get("error") or "").lower()
        if not res["success"] and ("disk" in err_lower or "space" in err_lower or "no space" in err_lower):
            clear_huggingface_cache()

    # сводка
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ГЕНЕРАЦИЙ")
    print("=" * 60)
    for r in results:
        status = "OK" if r["success"] else "FAIL"
        err = f" | {r['error']}" if r.get("error") else ""
        load = r.get("load_time_s")
        gen = r.get("gen_time_s")
        shape = r.get("audio_shape")
        path = r.get("path", "")
        print(f"  {r['template']}: {status} (load={load}s, gen={gen}s, shape={shape}){err}")
        if r["success"] and path:
            print(f"    -> {path}")
    print("=" * 60)
    ok = sum(1 for r in results if r["success"])
    print(f"Успешно: {ok}/{len(results)}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
