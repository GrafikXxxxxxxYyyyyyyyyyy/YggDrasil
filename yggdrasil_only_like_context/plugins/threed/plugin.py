"""3D structure generation plugin."""
from __future__ import annotations

from typing import Dict, Any

from ...plugins.base import AbstractPlugin


class ThreeDPlugin(AbstractPlugin):
    """3D generation plugin.
    
    Supports Gaussian Splatting, point cloud, and mesh diffusion.
    3D data is encoded into structured latents.
    """
    
    name = "3d"
    modality = "3d"
    description = "3D generation (Gaussian Splatting, point clouds, meshes)"
    version = "1.0.0"
    
    default_config = {
        "type": "model/modular",
        "backbone": {
            "type": "backbone/dit",
            "hidden_dim": 768,
            "num_layers": 12,
        },
        "codec": {
            "type": "codec/identity",
        },
        "conditioner": [
            {"type": "conditioner/clip_text"}
        ],
        "guidance": [{"type": "guidance/cfg", "scale": 7.5}],
        "diffusion_process": {"type": "diffusion/process/ddpm"},
    }
    
    @classmethod
    def register_blocks(cls):
        pass
    
    @classmethod
    def get_ui_schema(cls) -> Dict[str, Any]:
        return {
            "inputs": [
                {"type": "text", "name": "prompt", "label": "3D Object Description",
                 "placeholder": "A detailed 3D model of a chair..."},
                {"type": "image", "name": "reference_image", "label": "Reference Image",
                 "optional": True},
            ],
            "outputs": [
                {"type": "3d", "name": "result", "label": "Generated 3D"},
                {"type": "image", "name": "preview", "label": "Preview Render"},
            ],
            "advanced": [
                {"type": "slider", "name": "num_steps", "label": "Steps",
                 "min": 1, "max": 200, "default": 100},
                {"type": "slider", "name": "guidance_scale", "label": "CFG Scale",
                 "min": 1.0, "max": 20.0, "default": 7.5, "step": 0.5},
                {"type": "dropdown", "name": "representation", "label": "3D Representation",
                 "options": ["gaussian_splatting", "point_cloud", "mesh", "nerf"]},
                {"type": "slider", "name": "num_points", "label": "Number of Points",
                 "min": 1000, "max": 100000, "default": 10000, "step": 1000},
            ],
        }
