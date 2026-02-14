"""YggDrasil Plugins -- auto-registration of all modalities."""

import pkgutil
import importlib
from pathlib import Path

from .base import AbstractPlugin, PluginRegistry

# Auto-import all plugin sub-packages (image, video, audio, etc.)
# Skips 'custom' (template) and 'base' (abstract)
_SKIP = {"custom", "base"}
for _, name, is_pkg in pkgutil.iter_modules(__path__):
    if name not in _SKIP and not name.startswith("_"):
        try:
            importlib.import_module(f"{__name__}.{name}")
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load plugin '{name}': {e}")


def discover_plugins():
    """Re-run plugin discovery. Call after installing new plugins."""
    for _, name, is_pkg in pkgutil.iter_modules(__path__):
        if name not in _SKIP and not name.startswith("_"):
            try:
                importlib.import_module(f"{__name__}.{name}")
            except Exception:
                pass


def list_plugins():
    """List all registered plugins."""
    return PluginRegistry.list_plugins()


def get_plugin(name: str):
    """Get a plugin by name."""
    return PluginRegistry.get(name)