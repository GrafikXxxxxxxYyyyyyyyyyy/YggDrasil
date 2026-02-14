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
    - имеет typed I/O ports (declare_io) для dataflow-графа
    - имеет slots — места для подключения других блоков (legacy, backward compat)
    - умеет себя собирать из конфига
    - поддерживает pre/post hooks
    - имеет process() для port-based execution
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
        """Automatically build all slots from config."""
        from yggdrasil.core.block.builder import BlockBuilder
        
        for slot_name, slot in self.slots.items():
            if slot_name not in self.config:
                continue
            child_config = self.config[slot_name]
            
            # Handle list configs for multiple-accept slots
            if isinstance(child_config, (list, tuple)):
                for item in child_config:
                    if isinstance(item, AbstractBlock):
                        self.attach_slot(slot_name, item)
                    elif isinstance(item, dict) and "type" in item:
                        child = BlockBuilder.build(item)
                        self.attach_slot(slot_name, child)
            elif isinstance(child_config, AbstractBlock):
                self.attach_slot(slot_name, child_config)
            elif isinstance(child_config, dict) and "type" in child_config:
                child = BlockBuilder.build(child_config)
                self.attach_slot(slot_name, child)
            else:
                try:
                    child = BlockBuilder.build(child_config)
                    self.attach_slot(slot_name, child)
                except Exception:
                    pass  # Skip non-block configs
    
    def attach_slot(self, slot_name: str, block: "AbstractBlock"):
        """Attach a block to a slot (Lego action)."""
        if slot_name not in self.slots:
            raise KeyError(f"Slot {slot_name} does not exist in {self.block_type}")
        
        slot = self.slots[slot_name]
        if isinstance(block, type):
            raise TypeError(f"Slot {slot_name}: received class {block.__name__}, expected an instance.")
        
        if not slot.check_compatible(block):
            raise TypeError(
                f"Block {getattr(block, 'block_type', type(block).__name__)} "
                f"is not compatible with slot {slot_name} "
                f"(expects {slot.accepts})"
            )
        
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
    
    # ==================== PORT SYSTEM (Dataflow Graph) ====================
    
    @classmethod
    def declare_io(cls) -> Dict[str, "Port"]:
        """Декларация всех I/O портов блока.
        
        Переопределяется в потомках. Возвращает dict вида::
        
            {
                "x": InputPort("x", spec=TensorSpec(ndim=4, space="latent")),
                "timestep": InputPort("timestep", data_type="scalar"),
                "output": OutputPort("output", spec=TensorSpec(ndim=4, space="latent")),
            }
        
        Порты используются для:
        - Валидации соединений в ComputeGraph
        - Автогенерации UI в Gradio
        - Документации
        """
        return {}
    
    def process(self, **port_inputs: Any) -> Dict[str, Any]:
        """Выполнить блок через порты (для dataflow-графа).
        
        Принимает именованные входы (ключи = имена входных портов),
        возвращает dict выходов (ключи = имена выходных портов).
        
        По умолчанию делегирует в ``_forward_impl``.
        Переопределите этот метод для полного контроля над port-based execution.
        """
        result = self._forward_impl(**port_inputs)
        # Если _forward_impl возвращает dict — прокидываем как есть
        if isinstance(result, dict):
            return result
        # Иначе оборачиваем в {"output": result}
        return {"output": result}
    
    def get_input_ports(self) -> Dict[str, "Port"]:
        """Получить только входные порты."""
        return {
            name: port for name, port in self.declare_io().items()
            if port.direction == "input"
        }
    
    def get_output_ports(self) -> Dict[str, "Port"]:
        """Получить только выходные порты."""
        return {
            name: port for name, port in self.declare_io().items()
            if port.direction == "output"
        }
    
    # ==================== HOOKS ====================
    
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