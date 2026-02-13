from __future__ import annotations

from omegaconf import OmegaConf, DictConfig, ListConfig
from pathlib import Path
from typing import Dict, Any, Union, Optional
import copy

from ...core.block.slot import Slot
from ...core.block.registry import list_blocks


def load_config(path: str | Path | DictConfig) -> DictConfig:
    """Загрузка конфига с поддержкой наследования."""
    if isinstance(path, (str, Path)):
        cfg = OmegaConf.load(path)
    else:
        cfg = path
    
    # Поддержка !include и _base_
    if "_base_" in cfg:
        base = load_config(cfg._base_)
        cfg = OmegaConf.merge(base, cfg)
        del cfg["_base_"]
    
    return OmegaConf.create(cfg)


def save_config(config: DictConfig, path: str | Path):
    """Сохранение конфига."""
    OmegaConf.save(config, path)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Слияние нескольких конфигов (последний побеждает)."""
    return OmegaConf.merge(*configs)


def validate_slots(config: DictConfig, block_type: str) -> None:
    """Валидация слотов перед сборкой (очень полезно в runtime)."""
    from ...core.block.registry import get_block_class
    
    BlockClass = get_block_class(block_type)
    expected_slots = BlockClass._define_slots() if hasattr(BlockClass, "_define_slots") else {}
    
    for slot_name, slot in expected_slots.items():
        if not slot.optional and slot_name not in config:
            raise ValueError(f"Слот '{slot_name}' обязателен для {block_type}")
        
        if slot_name in config and "type" in config[slot_name]:
            # Рекурсивная проверка вложенных блоков
            child_type = config[slot_name].get("type")
            validate_slots(config[slot_name], child_type)