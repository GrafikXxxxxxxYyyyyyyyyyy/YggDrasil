#!/usr/bin/env python3
"""Список доступных аудио-шаблонов и всех пайплайнов с фильтром по модальности.

Запуск:
    python examples/audio/list_templates.py
    python examples/audio/list_templates.py --all
"""
import argparse
from yggdrasil.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Показать все шаблоны (не только audio)")
    args = parser.parse_args()

    if args.all:
        print("── Все доступные шаблоны (Pipeline.list_available) ──\n")
        available = Pipeline.list_available()
        for name, info in sorted(available.items()):
            mod = info.get("modality", "?")
            desc = (info.get("description") or "")[:70]
            print(f"  {name:35s} [{mod:6s}]  {desc}")
        return

    print("── Аудио-шаблоны (Pipeline.list_audio_templates) ──\n")
    for name, desc in Pipeline.list_audio_templates().items():
        print(f"  {name:35s}  {desc}")
    print("\nИспользование:")
    print("  pipe = Pipeline.from_template('musicldm_txt2audio', device='cuda')")
    print("  pipe = Pipeline.from_pretrained('cvssp/audioldm2')")


if __name__ == "__main__":
    main()
