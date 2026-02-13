from typing import Type, Optional, Any
from dataclasses import dataclass

from .base import AbstractBlock


@dataclass
class Slot:
    """Место для подключения другого блока (Lego-дырка)."""
    name: str
    accepts: Type[AbstractBlock] | tuple[Type[AbstractBlock], ...]   # Какие типы можно воткнуть
    multiple: bool = False          # Можно ли несколько (например, несколько адаптеров)
    optional: bool = False
    default: Optional[dict] = None  # Конфиг по умолчанию, если не указан
    
    def accepts(self, block: AbstractBlock) -> bool:
        """Проверка совместимости."""
        if isinstance(self.accepts, tuple):
            return isinstance(block, self.accepts)
        return isinstance(block, self.accepts)