import importlib
import logging
import pkgutil
from typing import Dict, List, Optional, Type, Any
from pathlib import Path
from difflib import get_close_matches

from .base import AbstractBaseBlock

logger = logging.getLogger(__name__)


class BlockRegistry:
    """Global registry of all Lego blocks.
    
    Blocks register themselves via the @register_block decorator.
    Auto-discovery scans yggdrasil.blocks and yggdrasil.plugins
    packages, logging any import failures (not silently swallowing them).
    """
    _registry: Dict[str, Type[AbstractBaseBlock]] = {}
    _import_errors: Dict[str, str] = {}  # module -> error message
    _discovered: bool = False
    
    @classmethod
    def register(cls, block_type: str):
        """Decorator: @register_block("model/modular")"""
        def decorator(block_cls: Type[AbstractBaseBlock]):
            cls._registry[block_type] = block_cls
            block_cls.block_type = block_type
            return block_cls
        return decorator
    
    @classmethod
    def get(cls, key: str) -> Type[AbstractBaseBlock]:
        """Get block class by key. Raises informative KeyError with suggestions."""
        if key not in cls._registry:
            cls._auto_discover()
            if key not in cls._registry:
                # Suggest similar block types
                similar = get_close_matches(key, cls._registry.keys(), n=5, cutoff=0.4)
                msg = f"Block '{key}' not found in registry."
                if similar:
                    msg += f" Did you mean: {', '.join(similar)}?"
                msg += f"\n  Available blocks ({len(cls._registry)}): {sorted(cls._registry.keys())[:20]}"
                if len(cls._registry) > 20:
                    msg += f" ... and {len(cls._registry) - 20} more"
                if cls._import_errors:
                    msg += f"\n  Note: {len(cls._import_errors)} modules failed to import during auto-discovery."
                raise KeyError(msg)
        return cls._registry[key]
    
    @classmethod
    def list_blocks(cls) -> Dict[str, Type[AbstractBaseBlock]]:
        cls._auto_discover()
        return cls._registry.copy()
    
    @classmethod
    def get_import_errors(cls) -> Dict[str, str]:
        """Return a dict of module -> error for all failed auto-discovery imports."""
        return cls._import_errors.copy()
    
    @classmethod
    def _auto_discover(cls):
        """Auto-import all blocks from blocks/ and plugins/ (recursive).
        
        Logs import failures so users can diagnose missing dependencies
        rather than silently hiding them.
        """
        if cls._discovered:
            return
        cls._discovered = True
        
        for package in ["yggdrasil.blocks", "yggdrasil.plugins"]:
            try:
                pkg = importlib.import_module(package)
                for importer, name, ispkg in pkgutil.walk_packages(
                    pkg.__path__, prefix=f"{package}."
                ):
                    try:
                        importlib.import_module(name)
                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {e}"
                        cls._import_errors[name] = error_msg
                        logger.debug(f"Auto-discover: failed to import {name}: {error_msg}")
            except ImportError:
                pass  # Package not installed — that's ok
            except Exception as e:
                logger.debug(f"Auto-discover: failed to scan {package}: {e}")
    
    @classmethod
    def reset(cls):
        """Reset discovery state (for testing)."""
        cls._discovered = False
        cls._import_errors.clear()


# Convenience aliases
register_block = BlockRegistry.register
get_block_class = BlockRegistry.get
list_blocks = BlockRegistry.list_blocks


def auto_discover():
    """Auto-import all modules from core/, blocks/, and training/ to register blocks.
    
    Uses walk_packages for recursive discovery.
    Logs import failures instead of silently swallowing them.
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
                    except Exception as e:
                        logger.debug(f"auto_discover: failed to import {name}: {type(e).__name__}: {e}")
        except ImportError:
            pass  # Package not installed
        except Exception as e:
            logger.debug(f"auto_discover: failed to scan {root}: {e}")