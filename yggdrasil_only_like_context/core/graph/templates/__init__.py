"""InferencePipeline templates — готовые графы для популярных пайплайнов.

Каждый template — функция, возвращающая ComputeGraph.
"""
from __future__ import annotations

from typing import Callable, Dict

# Registry of template builders
_TEMPLATE_REGISTRY: Dict[str, Callable] = {}


def register_template(name: str):
    """Декоратор для регистрации template builder."""
    def decorator(fn: Callable) -> Callable:
        _TEMPLATE_REGISTRY[name] = fn
        return fn
    return decorator


def get_template(name: str) -> Callable:
    """Получить template builder по имени."""
    # Lazy imports to register templates
    _ensure_templates_loaded()
    
    if name not in _TEMPLATE_REGISTRY:
        available = list(_TEMPLATE_REGISTRY.keys())
        raise KeyError(
            f"Template '{name}' not found. Available: {available}"
        )
    return _TEMPLATE_REGISTRY[name]


def list_templates() -> list[str]:
    """Список доступных шаблонов."""
    _ensure_templates_loaded()
    return sorted(_TEMPLATE_REGISTRY.keys())


def _ensure_templates_loaded():
    """Lazy-загрузка всех модулей с шаблонами."""
    if _TEMPLATE_REGISTRY:
        return
    
    import importlib
    modules = [
        "yggdrasil.core.graph.templates.image_pipelines",
        "yggdrasil.core.graph.templates.control_pipelines",
        "yggdrasil.core.graph.templates.video_pipelines",
        "yggdrasil.core.graph.templates.animatediff_extensions",
        "yggdrasil.core.graph.templates.audio_pipelines",
        "yggdrasil.core.graph.templates.specialized_pipelines",
        "yggdrasil.core.graph.templates.training_pipelines",
    ]
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError:
            pass  # Module may not exist yet
