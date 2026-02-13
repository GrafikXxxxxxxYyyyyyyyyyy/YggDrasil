from __future__ import annotations

from abc import ABC, abstractmethod
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Type

from ..core.block.registry import register_block
from ..core.model.modular import ModularDiffusionModel


class AbstractPlugin(ABC):
    """Базовый класс плагина-модальности.
    
    Каждый плагин (image, timeseries, molecular...) наследуется отсюда
    и регистрирует свои блоки.
    """
    
    name: str = "unknown"                    # "timeseries", "midi", "gaussian_3d"
    default_config: str | DictConfig = None  # путь к yaml или сам конфиг
    
    @classmethod
    @abstractmethod
    def register_blocks(cls):
        """Здесь регистрируются все блоки модальности."""
        pass
    
    @classmethod
    def get_default_model_config(cls) -> DictConfig:
        """Возвращает конфиг модели по умолчанию для этой модальности."""
        if isinstance(cls.default_config, str):
            from ..core.utils.config import load_config
            return load_config(cls.default_config)
        return cls.default_config or {}
    
    @classmethod
    def create_model(cls, **overrides) -> ModularDiffusionModel:
        """Быстрое создание модели этой модальности."""
        config = cls.get_default_model_config()
        config = OmegaConf.merge(config, overrides)
        from ..core.block.builder import BlockBuilder
        return BlockBuilder.build(config)