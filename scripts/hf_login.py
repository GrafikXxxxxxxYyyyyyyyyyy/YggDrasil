#!/usr/bin/env python3
"""Вход в Hugging Face без команды huggingface-cli.

Использование:
  python scripts/hf_login.py              # запросит токен в терминале
  python scripts/hf_login.py hf_xxx      # передать токен аргументом
  HF_TOKEN=hf_xxx python scripts/hf_login.py   # токен из переменной окружения

Токен можно создать: https://huggingface.co/settings/tokens
После входа примите лицензию модели, например:
  https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
"""
import os
import sys


def main():
    try:
        from huggingface_hub import login
    except ImportError:
        print("Установите: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    token = None
    if len(sys.argv) > 1:
        token = sys.argv[1].strip()
    if not token and os.environ.get("HF_TOKEN"):
        token = os.environ.get("HF_TOKEN").strip()

    if token:
        login(token=token)
        print("Вход выполнен, токен сохранён в кэш.")
    else:
        print("Вставьте токен с https://huggingface.co/settings/tokens (будут скрыты):")
        login()
        print("Вход выполнен.")


if __name__ == "__main__":
    main()
