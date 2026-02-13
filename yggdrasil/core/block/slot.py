# yggdrasil/core/block/slot.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Type, Optional, Any, Union


@dataclass
class Slot:
    """Место для подключения другого блока (Lego-дырка)."""
    name: str
    accepts: Union[Type["AbstractBlock"], tuple[Type["AbstractBlock"], ...], str]
    multiple: bool = False
    optional: bool = False
    default: Optional[dict] = None

    def accepts(self, block: "AbstractBlock") -> bool:
        """Проверка совместимости."""
        from .base import AbstractBlock
        
        if isinstance(self.accepts, str):
            return getattr(block, "block_type", "").startswith(self.accepts)
        
        if isinstance(self.accepts, tuple):
            return isinstance(block, self.accepts)
        if isinstance(block, type):
            return False
        cl = type(block)
        return self.accepts in cl.__mro__