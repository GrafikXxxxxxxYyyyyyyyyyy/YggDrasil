from __future__ import annotations

from abc import ABC, abstractmethod
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Type, List, Any, Optional

from ..core.block.registry import register_block
from ..core.model.modular import ModularDiffusionModel


class PluginRegistry:
    """Global registry of all loaded plugins."""
    _plugins: Dict[str, Type["AbstractPlugin"]] = {}
    
    @classmethod
    def register(cls, plugin_cls: Type["AbstractPlugin"]):
        cls._plugins[plugin_cls.name] = plugin_cls
    
    @classmethod
    def get(cls, name: str) -> Type["AbstractPlugin"]:
        if name not in cls._plugins:
            raise KeyError(f"Plugin '{name}' not found. Available: {list(cls._plugins.keys())}")
        return cls._plugins[name]
    
    @classmethod
    def list_plugins(cls) -> Dict[str, Type["AbstractPlugin"]]:
        return cls._plugins.copy()
    
    @classmethod
    def list_names(cls) -> List[str]:
        return list(cls._plugins.keys())


class AbstractPlugin(ABC):
    """Base class for modality plugins.
    
    Each plugin (image, audio, video, molecular, etc.) inherits from this
    and registers its blocks, configs, and UI schema.
    """
    
    name: str = "unknown"
    modality: str = "unknown"
    description: str = ""
    version: str = "1.0.0"
    default_config: str | DictConfig | dict = None
    
    def __init_subclass__(cls, **kwargs):
        """Auto-register subclasses."""
        super().__init_subclass__(**kwargs)
        if cls.name != "unknown":
            PluginRegistry.register(cls)
    
    @classmethod
    @abstractmethod
    def register_blocks(cls):
        """Register all blocks this plugin provides."""
        pass
    
    @classmethod
    def get_default_model_config(cls) -> DictConfig:
        """Return default model config for this modality."""
        if isinstance(cls.default_config, str):
            try:
                return OmegaConf.load(cls.default_config)
            except Exception:
                return OmegaConf.create({})
        if isinstance(cls.default_config, dict):
            return OmegaConf.create(cls.default_config)
        return cls.default_config or OmegaConf.create({})
    
    @classmethod
    def get_ui_schema(cls) -> Dict[str, Any]:
        """Return Gradio UI configuration for this modality.
        
        Override in subclass to provide modality-specific UI controls.
        
        Returns:
            Dict with keys: inputs, outputs, advanced
        """
        return {
            "inputs": [
                {"type": "text", "name": "prompt", "label": "Prompt"},
            ],
            "outputs": [
                {"type": "text", "name": "result", "label": "Result"},
            ],
            "advanced": [
                {"type": "slider", "name": "num_steps", "label": "Steps",
                 "min": 1, "max": 150, "default": 50},
                {"type": "slider", "name": "guidance_scale", "label": "Guidance Scale",
                 "min": 1.0, "max": 20.0, "default": 7.5, "step": 0.5},
                {"type": "dropdown", "name": "solver", "label": "Solver",
                 "options": ["ddim", "euler", "heun"]},
            ],
        }
    
    @classmethod
    def get_available_configs(cls) -> Dict[str, DictConfig]:
        """Return all available preset configs for this modality.
        
        Override to provide multiple presets (e.g. sd15, sdxl, flux).
        """
        return {"default": cls.get_default_model_config()}
    
    @classmethod
    def create_model(cls, config_name: str = "default", **overrides) -> ModularDiffusionModel:
        """Create a model for this modality."""
        configs = cls.get_available_configs()
        config = configs.get(config_name, cls.get_default_model_config())
        if overrides:
            config = OmegaConf.merge(config, overrides)
        from ..core.block.builder import BlockBuilder
        return BlockBuilder.build(config)