"""Template for creating a new modality plugin.

To create a new modality:
1. Copy the `custom/` directory to `your_modality/`
2. Rename this class and update name/modality/description
3. Implement register_blocks() to import your blocks
4. Override get_ui_schema() for custom UI controls
5. Update __init__.py to import your modality class
"""
from __future__ import annotations

import torch
from typing import Any, Dict
from omegaconf import DictConfig

from ...plugins.base import AbstractPlugin
from ...core.block.registry import register_block


class CustomModality(AbstractPlugin):
    """Template modality plugin.
    
    Replace "Custom" with your modality name (e.g. TimeseriesPlugin, MidiPlugin).
    """
    
    name = "custom"
    modality = "custom"
    description = "Template for custom modalities"
    version = "1.0.0"
    
    default_config = {
        "type": "model/modular",
        "backbone": {"type": "backbone/custom"},
        "codec": {"type": "codec/custom"},
        "diffusion_process": {"type": "diffusion/process/flow/rectified"},
    }
    
    @classmethod
    def register_blocks(cls):
        """Register all blocks for this modality."""
        from .backbone import CustomBackbone
        from .codec import CustomCodec
    
    @classmethod
    def get_ui_schema(cls) -> Dict[str, Any]:
        """Define Gradio UI controls for this modality."""
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
            ],
        }
    
    @staticmethod
    def to_tensor(data: Any) -> torch.Tensor:
        """Convert raw data to tensor."""
        return torch.as_tensor(data).float()
    
    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> Any:
        """Convert tensor back to user format."""
        return tensor.cpu().numpy()