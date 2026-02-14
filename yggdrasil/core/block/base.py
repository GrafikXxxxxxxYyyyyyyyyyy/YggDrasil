# yggdrasil/core/block/base.py
from __future__ import annotations

import warnings
import torch
import torch.nn as nn
from abc import ABC
from typing import Any, Dict, Optional, List
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


class AbstractBlock(ABC, nn.Module):
    """Базовый Lego-кирпичик всего фреймворка.
    
    Контракт блока (два обязательных метода):
    
    1. ``declare_io()`` — декларация типизированных портов ввода/вывода.
    2. ``process(**port_inputs) -> dict`` — обработка данных.
    
    Пример минимального блока::
    
        @register_block("my/super_res")
        class SuperRes(AbstractBlock):
            block_type = "my/super_res"
            
            def __init__(self, config=None):
                super().__init__(config or {"type": "my/super_res"})
                self.model = ...
            
            @classmethod
            def declare_io(cls):
                return {
                    "low_res": InputPort("low_res"),
                    "high_res": OutputPort("high_res"),
                }
            
            def process(self, **kw):
                return {"high_res": self.model(kw["low_res"])}
    
    Legacy-методы (``_forward_impl``, ``_define_slots``) поддерживаются
    для обратной совместимости, но новый код должен использовать
    только ``declare_io`` + ``process``.
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
        
        # Slots (legacy, backward compat)
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
        """Base forward with hooks. Delegates to process() for port-based execution."""
        for hook in self.pre_hooks:
            result = hook(self, *args, **kwargs)
            if result is not None:
                args, kwargs = result if isinstance(result, tuple) else (result, kwargs)
        
        # Call chain: forward() -> _forward_impl() (for legacy nn.Module compat)
        output = self._forward_impl(*args, **kwargs)
        
        for hook in self.post_hooks:
            result = hook(self, output, *args, **kwargs)
            if result is not None:
                output = result
        
        return output
    
    def _forward_impl(self, *args, **kwargs) -> Any:
        """Legacy forward implementation.
        
        Default: delegates to process() so that blocks only need to implement
        process() and both forward() and process() work correctly.
        
        Override this ONLY if you need raw nn.Module forward() compatibility
        (e.g., wrapping a diffusers model that expects positional args).
        For new blocks, override process() instead.
        """
        # Delegate to process() — this breaks the old circular dependency.
        # Old chain was: process() -> _forward_impl() -> NotImplementedError
        # New chain is:  _forward_impl() -> process() (if not overridden)
        #                process() is the single source of truth.
        result = self.process(**kwargs)
        if isinstance(result, dict) and "output" in result:
            return result["output"]
        return result
    
    # ==================== PORT SYSTEM (Dataflow Graph) ====================
    
    @classmethod
    def declare_io(cls) -> Dict[str, "Port"]:
        """Declare all I/O ports of this block.
        
        Override in subclasses. Return a dict like::
        
            {
                "x": InputPort("x", spec=TensorSpec(ndim=4, space="latent")),
                "timestep": InputPort("timestep", data_type="scalar"),
                "output": OutputPort("output", spec=TensorSpec(ndim=4, space="latent")),
            }
        
        Ports are used for:
        - Connection validation in ComputeGraph
        - Auto-generating UI in Gradio
        - Documentation
        """
        return {}
    
    def process(self, **port_inputs: Any) -> Dict[str, Any]:
        """Execute the block via named ports (dataflow graph API).
        
        This is THE primary method to implement for any new block.
        
        Accepts named inputs (keys = input port names),
        returns dict of outputs (keys = output port names).
        
        The default raises NotImplementedError — every block must implement
        either process() or _forward_impl().
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement process() "
            f"(preferred) or _forward_impl() (legacy). "
            f"See docs/custom_blocks.md for examples."
        )
    
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