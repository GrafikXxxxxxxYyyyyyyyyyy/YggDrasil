"""Text diffusion plugin (discrete/continuous text diffusion)."""
from __future__ import annotations

from typing import Dict, Any

from ...plugins.base import AbstractPlugin


class TextPlugin(AbstractPlugin):
    """Text generation via diffusion.
    
    Supports continuous-space (Diffusion-LM) and discrete-space (MDLM, D3PM)
    text diffusion models.
    """
    
    name = "text"
    modality = "text"
    description = "Text generation via diffusion (Diffusion-LM, MDLM)"
    version = "1.0.0"
    
    default_config = {
        "type": "model/modular",
        "backbone": {
            "type": "backbone/transformer_1d",
            "in_channels": 768,
            "hidden_dim": 768,
            "num_layers": 12,
        },
        "codec": {
            "type": "codec/identity",
        },
        "diffusion_process": {"type": "diffusion/process/ddpm"},
    }
    
    @classmethod
    def register_blocks(cls):
        pass
    
    @classmethod
    def get_ui_schema(cls) -> Dict[str, Any]:
        return {
            "inputs": [
                {"type": "text", "name": "prompt", "label": "Prompt / Prefix",
                 "placeholder": "Once upon a time..."},
                {"type": "text", "name": "constraint", "label": "Constraint",
                 "placeholder": "Must include the word 'dragon'", "optional": True},
            ],
            "outputs": [
                {"type": "text", "name": "result", "label": "Generated Text"},
            ],
            "advanced": [
                {"type": "slider", "name": "num_steps", "label": "Steps",
                 "min": 1, "max": 500, "default": 200},
                {"type": "slider", "name": "max_length", "label": "Max Length (tokens)",
                 "min": 16, "max": 1024, "default": 128},
                {"type": "slider", "name": "guidance_scale", "label": "CFG Scale",
                 "min": 1.0, "max": 10.0, "default": 3.0, "step": 0.5},
                {"type": "dropdown", "name": "diffusion_type", "label": "Diffusion Type",
                 "options": ["continuous", "discrete"]},
            ],
        }
