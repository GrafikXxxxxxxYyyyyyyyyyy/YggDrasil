"""
Block Registry: block_type -> class; build from config.

Canon: WorldGenerator_2.0/TODO_01 §5.
- register(block_type, class), build(config) -> instance.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type, Union

from yggdrasill.foundation.block import AbstractBaseBlock


class BlockRegistry:
    """
    Maps block_type (str) to block class or factory.
    build(config) creates instance using config["block_type"] (or config["type"]).
    """

    _global: Optional["BlockRegistry"] = None

    def __init__(self) -> None:
        self._builders: Dict[str, Union[Type[AbstractBaseBlock], Callable[..., AbstractBaseBlock]]] = {}

    @classmethod
    def global_registry(cls) -> BlockRegistry:
        if cls._global is None:
            cls._global = cls()
        return cls._global

    def register(
        self,
        block_type: str,
        block_class: Union[Type[AbstractBaseBlock], Callable[..., AbstractBaseBlock]],
    ) -> None:
        if not block_type or not block_type.strip():
            raise ValueError("block_type must be non-empty")
        self._builders[block_type.strip()] = block_class

    def get(self, block_type: str) -> Optional[Union[Type[AbstractBaseBlock], Callable[..., AbstractBaseBlock]]]:
        return self._builders.get(block_type)

    def build(self, config: Dict[str, Any]) -> AbstractBaseBlock:
        """
        Create block from config. Config must contain "block_type" or "type".
        Rest of config passed to constructor (block_id from config if present).
        """
        block_type = config.get("block_type") or config.get("type")
        if block_type is None:
            raise KeyError("config must contain 'block_type' or 'type'")
        builder = self._builders.get(block_type)
        if builder is None:
            available = ", ".join(sorted(self._builders))
            raise KeyError(f"Unknown block_type: {block_type!r}. Registered: {available}")
        # Build: pass config without type keys to block
        rest = {k: v for k, v in config.items() if k not in ("block_type", "type")}
        block_id = rest.pop("block_id", None)
        return builder(block_id=block_id, config=rest)

    def __contains__(self, block_type: str) -> bool:
        return block_type in self._builders


def register_block(block_type: str, registry: Optional[BlockRegistry] = None):
    """Decorator: register a block class under block_type."""
    reg = registry or BlockRegistry.global_registry()

    def decorator(cls: Type[AbstractBaseBlock]) -> Type[AbstractBaseBlock]:
        reg.register(block_type, cls)
        return cls
    return decorator
