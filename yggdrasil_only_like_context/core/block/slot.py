# yggdrasil/core/block/slot.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Type, Optional, Any, Union


@dataclass
class Slot:
    """Connection point for attaching blocks (Lego socket)."""
    name: str
    accepts: Union[Type["AbstractBaseBlock"], tuple[Type["AbstractBaseBlock"], ...], str]
    multiple: bool = False
    optional: bool = False
    default: Optional[dict] = None

    def check_compatible(self, block: "AbstractBaseBlock") -> bool:
        """Check if a block is compatible with this slot."""
        from .base import AbstractBaseBlock
        
        if isinstance(block, type):
            return False
        
        if isinstance(self.accepts, str):
            return getattr(block, "block_type", "").startswith(self.accepts)
        
        if isinstance(self.accepts, tuple):
            return isinstance(block, self.accepts)
        
        # Check isinstance (handles ABC and concrete classes)
        return isinstance(block, self.accepts)