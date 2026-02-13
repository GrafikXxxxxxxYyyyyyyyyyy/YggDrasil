from __future__ import annotations

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, List
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from .slot import Slot
from .builder import BlockBuilder


class AbstractBlock(ABC, nn.Module):
    """Базовый Lego-кирпичик всего фреймворка.
    
    Каждый блок:
    - имеет уникальный id
    - имеет slots — места для подключения других блоков
    - умеет себя собирать из конфига
    - поддерживает pre/post hooks
    """
    
    # Метаданные блока (заполняются при регистрации)
    block_type: str = "unknown"      # Например: "model/modular", "diffusion/process/ddpm"
    block_version: str = "1.0.0"
    
    def __init__(self, config: DictConfig | dict):
        super().__init__()
        self.config = OmegaConf.create(config) if isinstance(config, dict) else config
        self.block_id = self.config.get("id", f"{self.block_type}_{id(self)}")
        
        # Slots — это Lego-дырки, куда можно воткнуть другие блоки
        self.slots: Dict[str, Slot] = self._define_slots()
        
        # Дети (подключённые блоки)
        self.children: Dict[str, AbstractBlock] = {}
        
        # Hooks (для кастомного поведения без наследования)
        self.pre_hooks: List[callable] = []
        self.post_hooks: List[callable] = []
        
        self._build_slots()
    
    @abstractmethod
    def _define_slots(self) -> Dict[str, Slot]:
        """Здесь потомок определяет, какие слоты у него есть."""
        return {}
    
    def _build_slots(self):
        """Автоматически собирает все слоты из конфига."""
        for slot_name, slot in self.slots.items():
            if slot_name in self.config:
                child_config = self.config[slot_name]
                child = BlockBuilder.build(child_config)
                self.attach_slot(slot_name, child)
    
    def attach_slot(self, slot_name: str, block: AbstractBlock):
        """Подключить блок в слот (Lego-действие)."""
        if slot_name not in self.slots:
            raise KeyError(f"Слот {slot_name} не существует в {self.block_type}")
        
        slot = self.slots[slot_name]
        if not slot.accepts(block):
            raise TypeError(f"Блок {block.block_type} не подходит для слота {slot_name}")
        
        if slot.multiple:
            if slot_name not in self.children:
                self.children[slot_name] = []
            self.children[slot_name].append(block)
        else:
            self.children[slot_name] = block
        
        # Регистрируем в nn.Module для корректного .to(device)
        if isinstance(self.children[slot_name], list):
            for b in self.children[slot_name]:
                self.add_module(f"{slot_name}_{id(b)}", b)
        else:
            self.add_module(slot_name, block)
    
    def forward(self, *args, **kwargs) -> Any:
        """Базовый forward с хуками."""
        for hook in self.pre_hooks:
            args, kwargs = hook(self, *args, **kwargs)
        
        output = self._forward_impl(*args, **kwargs)
        
        for hook in self.post_hooks:
            output = hook(self, output, *args, **kwargs)
        
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
        """Сохранение блока + всех детей."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "weights.pt")
        OmegaConf.save(self.config, path / "config.yaml")
        # TODO: рекурсивное сохранение детей
    
    @classmethod
    def load(cls, path: Path | str) -> AbstractBlock:
        path = Path(path)
        config = OmegaConf.load(path / "config.yaml")
        instance = cls(config)
        instance.load_state_dict(torch.load(path / "weights.pt"))
        return instance
    
    def __repr__(self):
        return f"<{self.block_type} id={self.block_id}>"