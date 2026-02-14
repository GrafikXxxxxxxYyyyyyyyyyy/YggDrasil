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
        """Автоматически импортирует все блоки из blocks/ и plugins/ (рекурсивно)."""
        for package in ["yggdrasil.blocks", "yggdrasil.plugins"]:
            try:
                pkg = importlib.import_module(package)
                for importer, name, ispkg in pkgutil.walk_packages(
                    pkg.__path__, prefix=f"{package}."
                ):
                    try:
                        importlib.import_module(name)
                    except Exception:
                        pass
            except Exception:
                pass  # Пакет может отсутствовать


# Удобные алиасы
register_block = BlockRegistry.register
get_block_class = BlockRegistry.get
list_blocks = BlockRegistry.list_blocks


def auto_discover():
    """Auto-import all modules from core/ and blocks/ to register blocks.
    
    Uses walk_packages for recursive discovery.
    """
    root_packages = [
        "yggdrasil.core",
        "yggdrasil.blocks",
        "yggdrasil.training",
    ]
    
    for root in root_packages:
        try:
            pkg = importlib.import_module(root)
            if hasattr(pkg, "__path__"):
                for _, name, _ in pkgutil.walk_packages(pkg.__path__, prefix=f"{root}."):
                    try:
                        importlib.import_module(name)
                    except Exception:
                        pass
        except Exception:
            pass  # Package may not exist