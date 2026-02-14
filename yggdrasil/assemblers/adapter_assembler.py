"""AdapterAssembler -- attach/detach adapters to models."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
from omegaconf import DictConfig, OmegaConf

from ..core.model.modular import ModularDiffusionModel
from ..core.block.base import AbstractBlock
from ..core.block.builder import BlockBuilder
from ..core.block.registry import auto_discover


class AdapterAssembler:
    """Utility for attaching and managing adapters on models.
    
    Supports LoRA, ControlNet, IP-Adapter, T2I-Adapter, and custom adapters.
    """
    
    @staticmethod
    def attach_lora(
        model: ModularDiffusionModel,
        lora_path: Optional[str] = None,
        rank: int = 4,
        alpha: float = 1.0,
        target_modules: Optional[List[str]] = None,
    ) -> ModularDiffusionModel:
        """Attach a LoRA adapter to the model.
        
        Args:
            model: Target model
            lora_path: Path to pre-trained LoRA weights (optional)
            rank: LoRA rank
            alpha: LoRA alpha scaling
            target_modules: Which modules to apply LoRA to
            
        Returns:
            Model with LoRA attached
        """
        auto_discover()
        
        config = {
            "type": "adapter/lora",
            "rank": rank,
            "alpha": alpha,
        }
        if target_modules:
            config["target_modules"] = target_modules
        
        adapter = BlockBuilder.build(config)
        
        if lora_path:
            import torch
            state = torch.load(lora_path, map_location="cpu", weights_only=True)
            adapter.load_state_dict(state, strict=False)
        
        model.attach_adapter(adapter)
        return model
    
    @staticmethod
    def attach_controlnet(
        model: ModularDiffusionModel,
        controlnet_path: Optional[str] = None,
        conditioning_scale: float = 1.0,
    ) -> ModularDiffusionModel:
        """Attach a ControlNet adapter.
        
        Args:
            model: Target model
            controlnet_path: Path or HF model ID for ControlNet weights
            conditioning_scale: ControlNet conditioning scale
            
        Returns:
            Model with ControlNet attached
        """
        auto_discover()
        
        config = {
            "type": "adapter/controlnet",
            "conditioning_scale": conditioning_scale,
        }
        if controlnet_path:
            config["pretrained"] = controlnet_path
        
        adapter = BlockBuilder.build(config)
        model.attach_adapter(adapter)
        return model
    
    @staticmethod
    def attach_ip_adapter(
        model: ModularDiffusionModel,
        adapter_path: Optional[str] = None,
        scale: float = 1.0,
    ) -> ModularDiffusionModel:
        """Attach an IP-Adapter.
        
        Args:
            model: Target model
            adapter_path: Path to IP-Adapter weights
            scale: IP-Adapter scale
            
        Returns:
            Model with IP-Adapter attached
        """
        auto_discover()
        
        config = {
            "type": "adapter/ip_adapter",
            "scale": scale,
        }
        if adapter_path:
            config["pretrained"] = adapter_path
        
        adapter = BlockBuilder.build(config)
        model.attach_adapter(adapter)
        return model
    
    @staticmethod
    def attach_from_config(
        model: ModularDiffusionModel,
        adapter_config: dict | DictConfig,
    ) -> ModularDiffusionModel:
        """Attach an adapter from config dict.
        
        Args:
            model: Target model
            adapter_config: Adapter configuration with 'type' key
            
        Returns:
            Model with adapter attached
        """
        auto_discover()
        adapter = BlockBuilder.build(adapter_config)
        model.attach_adapter(adapter)
        return model
    
    @staticmethod
    def merge_adapters(
        model: ModularDiffusionModel,
        adapters: List[Dict[str, Any]],
    ) -> ModularDiffusionModel:
        """Attach multiple adapters to a model.
        
        Args:
            model: Target model
            adapters: List of adapter configs, each with 'type' key
            
        Returns:
            Model with all adapters attached
        """
        for adapter_cfg in adapters:
            AdapterAssembler.attach_from_config(model, adapter_cfg)
        return model
    
    @staticmethod
    def list_adapters(model: ModularDiffusionModel) -> List[str]:
        """List all adapters currently attached to a model."""
        adapters = model._slot_children.get("adapters", [])
        return [
            f"{getattr(a, 'block_type', type(a).__name__)} (id={getattr(a, 'block_id', '?')})"
            for a in adapters
        ]
