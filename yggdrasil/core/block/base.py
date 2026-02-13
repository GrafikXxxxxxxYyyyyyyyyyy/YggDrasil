# yggdrasil/core/block/base.py
from __future__ import annotations

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


class AbstractBlock(ABC, nn.Module):
    """Базовый Lego-кирпичик всего фреймворка.
    
    Каждый блок:
    - имеет уникальный id
    - имеет slots — места для подключения других блоков
    - умеет себя собирать из конфига
    - поддерживает pre/post hooks
    """
    
    # Метаданные блока (заполняются при регистрации)
    block_type: str = "unknown"
    block_version: str = "1.0.0"
    
    def __init__(self, config: DictConfig | dict):
        # Сначала nn.Module, чтобы _modules существовал при attach_slot -> add_module в _build_slots
        nn.Module.__init__(self)
        super().__init__()
        self.config = OmegaConf.create(config) if isinstance(config, dict) else config
        self.block_id = self.config.get("id", f"{self.block_type}_{id(self)}")
        
        # Slots — это Lego-дырки, куда можно воткнуть другие блоки
        self.slots: Dict[str, "Slot"] = self._define_slots()
        
        # Дети (подключённые блоки). Имя _slot_children, чтобы не перекрывать nn.Module.children()
        self._slot_children: Dict[str, Any] = {}
        
        # Hooks (для кастомного поведения без наследования)
        self.pre_hooks: List[callable] = []
        self.post_hooks: List[callable] = []
        
        self._build_slots()
    
    def _define_slots(self) -> Dict[str, "Slot"]:
        """Здесь потомок определяет, какие слоты у него есть.
        Импортируем Slot внутри метода, чтобы избежать circular import."""
        from .slot import Slot
        return {}
    
    def _build_slots(self):
        """Автоматически собирает все слоты из конфига."""
        from yggdrasil.core.block.builder import BlockBuilder
        
        for slot_name, slot in self.slots.items():
            if slot_name not in self.config:
                continue
            child_config = self.config[slot_name]
            # Уже собранный блок (например, после BlockBuilder._resolve_slots)
            if isinstance(child_config, AbstractBlock):
                child = child_config
            else:
                child = BlockBuilder.build(child_config)
            self.attach_slot(slot_name, child)
    
    def attach_slot(self, slot_name: str, block: "AbstractBlock"):
        """Подключить блок в слот (Lego-действие)."""
        if slot_name not in self.slots:
            raise KeyError(f"Слот {slot_name} не существует в {self.block_type}")
        
        slot = self.slots[slot_name]
        if isinstance(block, type):
            raise TypeError(f"Слот {slot_name}: передан класс {block.__name__}, нужен экземпляр.")
        # Быстрая проверка по block_type без вызова slot.accepts (избегаем рекурсии/ABC)
        block_cl = type(block)
        ok = getattr(slot.accepts, "__mro__", None) and slot.accepts in block_cl.__mro__
        if not ok and hasattr(block, "block_type"):
            ok = getattr(slot.accepts, "block_type", None) and block.block_type == getattr(slot.accepts, "block_type", "")
        if not ok:
            raise TypeError(f"Блок {block.block_type} не подходит для слота {slot_name}")
        
        if slot.multiple:
            if slot_name not in self._slot_children:
                self._slot_children[slot_name] = []
            self._slot_children[slot_name].append(block)
        else:
            self._slot_children[slot_name] = block
        
        # Регистрируем в nn.Module для корректного .to(device)
        if isinstance(self._slot_children.get(slot_name), list):
            for i, b in enumerate(self._slot_children[slot_name]):
                self.add_module(f"{slot_name}_{i}", b)
        else:
            self.add_module(slot_name, block)
    
    def forward(self, *args, **kwargs) -> Any:
        """Базовый forward с хуками."""
        for hook in self.pre_hooks:
            result = hook(self, *args, **kwargs)
            if result is not None:
                args, kwargs = result if isinstance(result, tuple) else (result, kwargs)
        
        output = self._forward_impl(*args, **kwargs)
        
        for hook in self.post_hooks:
            result = hook(self, output, *args, **kwargs)
            if result is not None:
                output = result
        
        return output
    
    @abstractmethod
    def _forward_impl(self, *args, **kwargs) -> Any:
        """Реальная реализация forward в потомке."""
        pass
    
    def add_pre_hook(self, hook: callable):
        self.pre_hooks.append(hook)
    
    def add_post_hook(self, hook: callable):
        self.post_hooks.append(hook)
    
    def save(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "weights.pt")
        OmegaConf.save(self.config, path / "config.yaml")
    
    @classmethod
    def load(cls, path: Path | str) -> "AbstractBlock":
        path = Path(path)
        config = OmegaConf.load(path / "config.yaml")
        instance = cls(config)
        instance.load_state_dict(torch.load(path / "weights.pt"))
        return instance
    
    def __repr__(self):
        return f"<{self.block_type} id={self.block_id}>"