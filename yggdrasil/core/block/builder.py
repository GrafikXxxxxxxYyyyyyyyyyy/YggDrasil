from omegaconf import DictConfig, OmegaConf
from typing import Any

from .base import AbstractBlock
from .registry import get_block_class


class BlockBuilder:
    """Главный сборщик Lego. Превращает YAML → живой граф блоков."""
    
    @classmethod
    def build(cls, config: DictConfig | dict | str) -> AbstractBlock:
        """Собрать блок из конфига."""
        if isinstance(config, str):
            config = OmegaConf.load(config)
        elif isinstance(config, dict):
            config = OmegaConf.create(config)
        
        block_type = config.get("type") or config.get("block_type")
        if not block_type:
            raise ValueError("В конфиге обязателен ключ 'type' или 'block_type'")
        
        block_type_str = str(block_type)
        BlockClass = get_block_class(block_type_str)
        
        # Для model/modular не подставляем детей в конфиг (OmegaConf не хранит объекты),
        # дети соберутся в _build_slots через BlockBuilder.build(child_config)
        if block_type == "model/modular":
            resolved_config = config
        else:
            resolved_config = cls._resolve_slots(config)
        
        return BlockClass(resolved_config)
    
    @classmethod
    def _resolve_slots(cls, config: DictConfig) -> DictConfig:
        """Рекурсивно разрешает все слоты."""
        config = OmegaConf.create(config)  # Копируем
        
        for key in list(config.keys()):
            if isinstance(config[key], (dict, DictConfig)) and "type" in config[key]:
                # Это вложенный блок
                child = cls.build(config[key])
                config[key] = child  # Заменяем конфиг на готовый объект
        
        return config