"""Video modality plugin."""
from __future__ import annotations

from typing import Dict, Any

from ...plugins.base import AbstractPlugin


class VideoPlugin(AbstractPlugin):
    """Video generation plugin.
    
    Video diffusion uses 3D latents (time, height, width).
    Supports CogVideoX, Hunyuan Video, and custom architectures.
    """
    
    name = "video"
    modality = "video"
    description = "Video generation (CogVideoX, Hunyuan, Mochi)"
    version = "1.0.0"
    
    default_config = {
        "type": "model/modular",
        "backbone": {
            "type": "backbone/dit",
            "hidden_dim": 1536,
            "num_layers": 30,
            "num_heads": 24,
        },
        "codec": {
            "type": "codec/autoencoder_kl",
            "latent_channels": 16,
            "spatial_scale_factor": 8,
        },
        "conditioner": [
            {"type": "conditioner/t5_text", "pretrained": "google/t5-v1_1-xxl"}
        ],
        "guidance": [{"type": "guidance/cfg", "scale": 6.0}],
        "diffusion_process": {"type": "diffusion/process/flow/rectified"},
    }
    
    @classmethod
    def register_blocks(cls):
        pass
    
    @classmethod
    def get_ui_schema(cls) -> Dict[str, Any]:
        return {
            "inputs": [
                {"type": "text", "name": "prompt", "label": "Video Description",
                 "placeholder": "A cat playing with a ball in slow motion..."},
                {"type": "image", "name": "first_frame", "label": "First Frame (optional)",
                 "optional": True},
            ],
            "outputs": [
                {"type": "video", "name": "result", "label": "Generated Video"},
            ],
            "advanced": [
                {"type": "slider", "name": "num_steps", "label": "Steps",
                 "min": 1, "max": 100, "default": 50},
                {"type": "slider", "name": "guidance_scale", "label": "CFG Scale",
                 "min": 1.0, "max": 15.0, "default": 6.0, "step": 0.5},
                {"type": "slider", "name": "num_frames", "label": "Number of Frames",
                 "min": 8, "max": 128, "default": 49, "step": 1},
                {"type": "slider", "name": "fps", "label": "FPS",
                 "min": 8, "max": 60, "default": 24, "step": 1},
                {"type": "slider", "name": "width", "label": "Width",
                 "min": 256, "max": 1280, "default": 720, "step": 16},
                {"type": "slider", "name": "height", "label": "Height",
                 "min": 256, "max": 1280, "default": 480, "step": 16},
            ],
        }
