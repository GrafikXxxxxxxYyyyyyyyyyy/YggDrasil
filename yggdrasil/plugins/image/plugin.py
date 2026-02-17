"""Image modality plugin -- the most common diffusion modality."""
from __future__ import annotations

from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf

from ...plugins.base import AbstractPlugin


class ImagePlugin(AbstractPlugin):
    """Image generation plugin.
    
    Provides preset configs for major image diffusion architectures:
    SD 1.5, SDXL, SD3, Flux.
    """
    
    name = "image"
    modality = "image"
    description = "Image generation (SD 1.5, SDXL, SD3, Flux, custom)"
    version = "1.0.0"
    
    # Default to SD 1.5
    default_config = {
        "type": "model/modular",
        "backbone": {
            "type": "backbone/unet2d_condition",
            "pretrained": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        },
        "codec": {
            "type": "codec/autoencoder_kl",
            "pretrained": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "latent_channels": 4,
            "spatial_scale_factor": 8,
        },
        "conditioner": [
            {
                "type": "conditioner/clip_text",
                "pretrained": "openai/clip-vit-large-patch14",
            }
        ],
        "guidance": [
            {"type": "guidance/cfg", "scale": 7.5}
        ],
        "diffusion_process": {
            "type": "diffusion/process/ddpm",
            "num_train_timesteps": 1000,
        },
    }
    
    @classmethod
    def register_blocks(cls):
        """Register all image-related blocks."""
        # These are already registered via @register_block decorators
        # Just ensure the modules are imported
        try:
            from ...blocks.backbones.unet_2d_condition import UNet2DConditionBackbone
        except ImportError:
            pass
        try:
            from ...blocks.codecs.autoencoder_kl import AutoencoderKLCodec
        except ImportError:
            pass
        try:
            from ...blocks.conditioners.clip_text import CLIPTextConditioner
        except ImportError:
            pass
        try:
            from ...blocks.conditioners.sd3_text import SD3TextConditioner
        except ImportError:
            pass
        try:
            from ...blocks.guidances.cfg import ClassifierFreeGuidance
        except ImportError:
            pass
        try:
            from ...blocks.conditioners.ip_adapter_mask import IPAdapterMaskConditioner
        except ImportError:
            pass
        try:
            from ...blocks.conditioners.faceid import FaceIDConditioner
        except ImportError:
            pass
        try:
            from ...blocks.adapters.ip_adapter_plus import IPAdapterPlus
        except ImportError:
            pass
        try:
            from ...blocks.adapters.ip_adapter_faceid import IPAdapterFaceID
        except ImportError:
            pass
        try:
            from ...blocks.adapters.ip_adapter_merge import IPAdapterMerge
        except ImportError:
            pass
    
    @classmethod
    def get_ui_schema(cls) -> Dict[str, Any]:
        return {
            "inputs": [
                {"type": "text", "name": "prompt", "label": "Prompt",
                 "placeholder": "A beautiful landscape painting..."},
                {"type": "text", "name": "negative_prompt", "label": "Negative Prompt",
                 "placeholder": "blurry, low quality", "optional": True},
                {"type": "image", "name": "init_image", "label": "Init Image (img2img)",
                 "optional": True},
                {"type": "image", "name": "control_image", "label": "Control Image",
                 "optional": True},
            ],
            "outputs": [
                {"type": "image", "name": "result", "label": "Generated Image"},
                {"type": "gallery", "name": "gallery", "label": "Gallery"},
            ],
            "advanced": [
                {"type": "slider", "name": "num_steps", "label": "Steps",
                 "min": 1, "max": 150, "default": 50},
                {"type": "slider", "name": "guidance_scale", "label": "CFG Scale",
                 "min": 1.0, "max": 20.0, "default": 7.5, "step": 0.5},
                {"type": "slider", "name": "width", "label": "Width",
                 "min": 256, "max": 2048, "default": 512, "step": 64},
                {"type": "slider", "name": "height", "label": "Height",
                 "min": 256, "max": 2048, "default": 512, "step": 64},
                {"type": "dropdown", "name": "solver", "label": "Solver",
                 "options": ["ddim", "euler", "heun", "dpm++"]},
                {"type": "dropdown", "name": "scheduler", "label": "Noise Schedule",
                 "options": ["linear", "cosine", "sigmoid"]},
                {"type": "number", "name": "seed", "label": "Seed",
                 "default": -1},
                {"type": "slider", "name": "strength", "label": "Denoising Strength (img2img)",
                 "min": 0.0, "max": 1.0, "default": 0.75, "step": 0.05},
            ],
        }
    
    @classmethod
    def get_available_configs(cls) -> Dict[str, dict]:
        """Return preset configs for major image architectures."""
        return {
            "sd15": {
                "type": "model/modular",
                "backbone": {
                    "type": "backbone/unet2d_condition",
                    "pretrained": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                },
                "codec": {
                    "type": "codec/autoencoder_kl",
                    "pretrained": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                    "latent_channels": 4,
                    "spatial_scale_factor": 8,
                },
                "conditioner": [
                    {"type": "conditioner/clip_text",
                     "pretrained": "openai/clip-vit-large-patch14"}
                ],
                "guidance": [{"type": "guidance/cfg", "scale": 7.5}],
                "diffusion_process": {
                    "type": "diffusion/process/ddpm",
                    "num_train_timesteps": 1000,
                },
            },
            "sdxl": {
                "type": "model/modular",
                "backbone": {
                    "type": "backbone/unet2d_condition",
                    "pretrained": "stabilityai/stable-diffusion-xl-base-1.0",
                },
                "codec": {
                    "type": "codec/autoencoder_kl",
                    "pretrained": "stabilityai/stable-diffusion-xl-base-1.0",
                    "latent_channels": 4,
                    "spatial_scale_factor": 8,
                },
                "conditioner": [
                    {"type": "conditioner/clip_text",
                     "pretrained": "openai/clip-vit-large-patch14"},
                    {"type": "conditioner/clip_text",
                     "pretrained": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"},
                ],
                "guidance": [{"type": "guidance/cfg", "scale": 7.5}],
                "diffusion_process": {
                    "type": "diffusion/process/ddpm",
                    "num_train_timesteps": 1000,
                },
            },
            "sd3": {
                "type": "model/modular",
                "backbone": {
                    "type": "backbone/mmdit",
                    "pretrained": "stabilityai/stable-diffusion-3-medium",
                },
                "codec": {
                    "type": "codec/autoencoder_kl",
                    "pretrained": "stabilityai/stable-diffusion-3-medium",
                    "latent_channels": 16,
                    "spatial_scale_factor": 8,
                },
                "conditioner": [
                    {"type": "conditioner/clip_text",
                     "pretrained": "openai/clip-vit-large-patch14"},
                    {"type": "conditioner/t5_text",
                     "pretrained": "google/t5-v1_1-xxl"},
                ],
                "guidance": [{"type": "guidance/cfg", "scale": 5.0}],
                "diffusion_process": {
                    "type": "diffusion/process/flow/rectified",
                },
            },
            "flux": {
                "type": "model/modular",
                "backbone": {
                    "type": "backbone/mmdit",
                    "pretrained": "black-forest-labs/FLUX.1-dev",
                },
                "codec": {
                    "type": "codec/autoencoder_kl",
                    "pretrained": "black-forest-labs/FLUX.1-dev",
                    "latent_channels": 16,
                    "spatial_scale_factor": 8,
                },
                "conditioner": [
                    {"type": "conditioner/clip_text",
                     "pretrained": "openai/clip-vit-large-patch14"},
                    {"type": "conditioner/t5_text",
                     "pretrained": "google/t5-v1_1-xxl"},
                ],
                "guidance": [{"type": "guidance/cfg", "scale": 3.5}],
                "diffusion_process": {
                    "type": "diffusion/process/flow/rectified",
                },
            },
        }
