"""YggDrasil Plugins — авто-регистрация всех модальностей."""

import pkgutil
import importlib
from pathlib import Path

# Автоматически импортируем все подпапки (image, video, custom, my_awesome_modality и т.д.)
for _, name, _ in pkgutil.iter_modules(__path__):
    if name != "custom":  # custom — это шаблон, его не импортируем автоматически
        importlib.import_module(f"{__name__}.{name}")