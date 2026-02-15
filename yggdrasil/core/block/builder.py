from omegaconf import DictConfig, OmegaConf
from typing import Any

from .base import AbstractBaseBlock
from .registry import get_block_class


class BlockBuilder:
    """Main Lego assembler. Turns YAML config -> live block graph."""
    
    @classmethod
    def build(cls, config: DictConfig | dict | str) -> AbstractBaseBlock:
        """Build a block from config.
        
        Raises:
            ValueError: If 'type' or 'block_type' is missing from config.
            KeyError: If block type not found (with suggestions for similar blocks).
        """
        if isinstance(config, str):
            config = OmegaConf.load(config)
        elif isinstance(config, dict):
            config = OmegaConf.create(config)
        
        block_type = config.get("type") or config.get("block_type")
        if not block_type:
            raise ValueError(
                f"Config must contain 'type' or 'block_type' key. "
                f"Got keys: {list(config.keys())}"
            )
        
        block_type_str = str(block_type)
        # get_block_class now raises informative KeyError with suggestions
        BlockClass = get_block_class(block_type_str)
        
        # For model/modular: don't substitute children in config (OmegaConf can't store objects);
        # children are built via _build_slots -> BlockBuilder.build(child_config)
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