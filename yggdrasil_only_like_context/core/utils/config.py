from __future__ import annotations

from omegaconf import OmegaConf, DictConfig, ListConfig
from pathlib import Path
from typing import Dict, Any, Union, Optional
import copy

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
    """Validate required config keys for known block types (graph engine; no slots)."""
    if block_type == "model/modular" and "backbone" not in config:
        raise ValueError("model/modular requires 'backbone' in config")