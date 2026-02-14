import importlib
import pkgutil
from typing import Dict, Type, Any
from pathlib import Path

from .base import AbstractBlock


class BlockRegistry:
    """Глобальный реестр всех Lego-кирпичиков."""
    _registry: Dict[str, Type[AbstractBlock]] = {}
    
    @classmethod
    def register(cls, block_type: str):
        """Декоратор: @register_block("model/modular")"""
        def decorator(block_cls: Type[AbstractBlock]):
            # Просто используем block_type как ключ
            cls._registry[block_type] = block_cls
            block_cls.block_type = block_type
            return block_cls
        return decorator
    
    @classmethod
    def get(cls, key: str) -> Type[AbstractBlock]:
        """Получить класс по ключу."""
        if key not in cls._registry:
            # Попытка автоимпорта
            cls._auto_discover()
            if key not in cls._registry:
                raise KeyError(f"Блок {key} не найден. Доступные: {list(cls._registry.keys())}")
        return cls._registry[key]
    
    @classmethod
    def list_blocks(cls) -> Dict[str, Type[AbstractBlock]]:
        cls._auto_discover()
        return cls._registry.copy()
    
    @classmethod
    def _auto_discover(cls):
        """Автоматически импортирует все блоки из blocks/ и plugins/."""
        for package in ["yggdrasil.blocks", "yggdrasil.plugins"]:
            try:
                package_path = importlib.import_module(package).__path__
                for _, name, _ in pkgutil.iter_modules(package_path):
                    importlib.import_module(f"{package}.{name}")
            except Exception:
                pass  # Пакет может отсутствовать


# Удобные алиасы
register_block = BlockRegistry.register
get_block_class = BlockRegistry.get
list_blocks = BlockRegistry.list_blocks


def auto_discover():
    """Auto-import all modules from core/ and blocks/ to register blocks."""
    packages = [
        "yggdrasil.core.block",
        "yggdrasil.core.model",
        "yggdrasil.core.diffusion",
        "yggdrasil.core.diffusion.solver",
        "yggdrasil.core.diffusion.noise",
        "yggdrasil.core.engine",
        "yggdrasil.blocks.adapters",
        "yggdrasil.blocks.backbones",
        "yggdrasil.blocks.codecs",
        "yggdrasil.blocks.conditioners",
        "yggdrasil.blocks.guidances",
        "yggdrasil.blocks.positions",
        "yggdrasil.training",
    ]
    
    for pkg in packages:
        try:
            package = importlib.import_module(pkg)
            if hasattr(package, "__path__"):
                for _, module_name, _ in pkgutil.iter_modules(package.__path__):
                    try:
                        importlib.import_module(f"{pkg}.{module_name}")
                    except Exception:
                        pass
        except Exception:
            pass  # Package may not exist