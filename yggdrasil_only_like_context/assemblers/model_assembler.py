"""ModelAssembler -- build ModularDiffusionModel from various sources."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
from omegaconf import DictConfig, OmegaConf

from ..core.model.modular import ModularDiffusionModel
from ..core.block.builder import BlockBuilder
from ..core.block.registry import auto_discover


class ModelAssembler:
    """High-level factory for creating ModularDiffusionModel instances.
    
    Supports construction from:
    - YAML config file
    - Dict/DictConfig
    - HuggingFace model ID (auto-detect architecture)
    - Diffusers pipeline object
    - Plugin preset name
    """
    
    @staticmethod
    def from_config(config: str | dict | DictConfig) -> ModularDiffusionModel:
        """Build model from YAML path or dict config.
        
        Args:
            config: Path to YAML file, dict, or DictConfig
            
        Returns:
            Assembled ModularDiffusionModel
        """
        auto_discover()
        
        if isinstance(config, (str, Path)):
            config = OmegaConf.load(str(config))
        elif isinstance(config, dict):
            config = OmegaConf.create(config)
        
        # If config has a 'model' key, extract it
        if "model" in config:
            config = config["model"]
        
        return BlockBuilder.build(config)
    
    @staticmethod
    def from_pretrained(
        model_id: str,
        torch_dtype=None,
        device: str = "auto",
        **kwargs
    ) -> ModularDiffusionModel:
        """Build model from HuggingFace model ID.
        
        Auto-detects the architecture (SD 1.5, SDXL, SD3, Flux, etc.)
        and constructs the appropriate YggDrasil model.
        
        Args:
            model_id: HuggingFace model ID (e.g. "stable-diffusion-v1-5/stable-diffusion-v1-5")
            torch_dtype: Optional torch dtype
            device: Device to load model on
            
        Returns:
            Assembled ModularDiffusionModel
        """
        auto_discover()
        
        # Try loading via diffusers integration
        from ..integration.diffusers import load_from_diffusers
        return load_from_diffusers(model_id, torch_dtype=torch_dtype, **kwargs)
    
    @staticmethod
    def from_diffusers(pipe, **kwargs) -> ModularDiffusionModel:
        """Convert a diffusers pipeline to YggDrasil ModularDiffusionModel.
        
        Args:
            pipe: A diffusers DiffusionPipeline instance
            
        Returns:
            Assembled ModularDiffusionModel
        """
        auto_discover()
        
        # Detect pipeline type and route accordingly
        pipe_class = type(pipe).__name__
        
        config = {
            "type": "model/modular",
        }
        
        # Extract components from diffusers pipeline
        if hasattr(pipe, "unet"):
            config["backbone"] = {
                "type": "backbone/unet2d_condition",
                "_diffusers_model": pipe.unet,
            }
        elif hasattr(pipe, "transformer"):
            config["backbone"] = {
                "type": "backbone/dit",
                "_diffusers_model": pipe.transformer,
            }
        
        if hasattr(pipe, "vae"):
            config["codec"] = {
                "type": "codec/autoencoder_kl",
                "_diffusers_model": pipe.vae,
            }
        
        if hasattr(pipe, "text_encoder"):
            config["conditioner"] = [{
                "type": "conditioner/clip_text",
                "_diffusers_model": pipe.text_encoder,
                "_diffusers_tokenizer": getattr(pipe, "tokenizer", None),
            }]
        
        config["guidance"] = [{"type": "guidance/cfg", "scale": 7.5}]
        
        return BlockBuilder.build(OmegaConf.create(config))
    
    @staticmethod
    def from_plugin(
        plugin_name: str,
        config_name: str = "default",
        **overrides
    ) -> ModularDiffusionModel:
        """Build model from a plugin preset.
        
        Args:
            plugin_name: Plugin name (e.g. "image", "audio")
            config_name: Config preset name (e.g. "sd15", "sdxl")
            **overrides: Config overrides
            
        Returns:
            Assembled ModularDiffusionModel
        """
        auto_discover()
        
        from ..plugins.base import PluginRegistry
        plugin = PluginRegistry.get(plugin_name)
        return plugin.create_model(config_name=config_name, **overrides)
    
    @staticmethod
    def from_recipe(recipe_name: str, **overrides) -> ModularDiffusionModel:
        """Build model from a recipe YAML.
        
        Args:
            recipe_name: Recipe name (e.g. "sd15_generate")
            
        Returns:
            Assembled ModularDiffusionModel
        """
        from ..configs import get_recipe
        config = get_recipe(recipe_name, **overrides)
        return ModelAssembler.from_config(config)
