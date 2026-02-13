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
            # Если передана строка — проверяем по типу блока
            return block.block_type.startswith(self.accepts)
        
        if isinstance(self.accepts, tuple):
            return isinstance(block, self.accepts)
        return isinstance(block, self.accepts)