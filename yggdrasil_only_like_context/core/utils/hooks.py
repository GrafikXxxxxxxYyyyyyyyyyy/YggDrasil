from __future__ import annotations

from typing import Callable, Dict, Any, List
from functools import wraps
import inspect

from ...core.block.base import AbstractBaseBlock


class HookRegistry:
    """Глобальный реестр хуков (pre/post) для любого блока."""
    
    _hooks: Dict[str, List[Callable]] = {
        "pre": {},
        "post": {}
    }
    
    @classmethod
    def register(cls, hook_type: str, block_id: str, func: Callable):
        """Регистрация хука."""
        if block_id not in cls._hooks[hook_type]:
            cls._hooks[hook_type][block_id] = []
        cls._hooks[hook_type][block_id].append(func)
    
    @classmethod
    def get_hooks(cls, hook_type: str, block_id: str) -> List[Callable]:
        return cls._hooks[hook_type].get(block_id, [])
    
    @classmethod
    def clear(cls):
        cls._hooks = {"pre": {}, "post": {}}


def register_hook(hook_type: str = "pre"):
    """Декоратор для удобной регистрации хуков."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Автоматически определяем блок
        frame = inspect.currentframe()
        while frame:
            if "self" in frame.f_locals:
                self = frame.f_locals["self"]
                if isinstance(self, AbstractBaseBlock):
                    block_id = getattr(self, "block_id", self.block_type)
                    HookRegistry.register(hook_type, block_id, func)
                    break
            frame = frame.f_back
        return wrapper
    return decorator


def apply_hooks(block: AbstractBaseBlock, hook_type: str, *args, **kwargs) -> Any:
    """Применить все зарегистрированные хуки."""
    for hook in HookRegistry.get_hooks(hook_type, block.block_id):
        args, kwargs = hook(block, *args, **kwargs) or (args, kwargs)
    return args, kwargs