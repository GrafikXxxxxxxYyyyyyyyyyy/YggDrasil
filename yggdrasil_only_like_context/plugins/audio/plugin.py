"""Audio modality plugin for music, speech, and sound effects."""
from __future__ import annotations

from typing import Dict, Any

from ...plugins.base import AbstractPlugin


class AudioPlugin(AbstractPlugin):
    """Audio generation plugin.
    
    Supports AudioLDM, Stable Audio, and custom audio diffusion architectures.
    Audio is typically encoded into 1D latents via Encodec or mel-spectrogram VAE.
    """
    
    name = "audio"
    modality = "audio"
    description = "Audio generation (music, speech, sound effects)"
    version = "1.0.0"
    
    default_config = {
        "type": "model/modular",
        "backbone": {
            "type": "backbone/transformer_1d",
            "in_channels": 64,
            "hidden_dim": 1024,
            "num_layers": 24,
        },
        "codec": {
            "type": "codec/encodec",
            "sample_rate": 16000,
            "latent_channels": 64,
        },
        "conditioner": [
            {"type": "conditioner/clap", "pretrained": "laion/clap-htsat-unfused"}
        ],
        "guidance": [{"type": "guidance/cfg", "scale": 4.5}],
        "diffusion_process": {"type": "diffusion/process/ddpm"},
    }
    
    @classmethod
    def register_blocks(cls):
        """Register audio-specific blocks."""
        pass  # Blocks registered via @register_block when modules are imported
    
    @classmethod
    def get_ui_schema(cls) -> Dict[str, Any]:
        return {
            "inputs": [
                {"type": "text", "name": "prompt", "label": "Audio Description",
                 "placeholder": "A jazz piano solo with drums..."},
                {"type": "audio", "name": "init_audio", "label": "Init Audio",
                 "optional": True},
            ],
            "outputs": [
                {"type": "audio", "name": "result", "label": "Generated Audio"},
            ],
            "advanced": [
                {"type": "slider", "name": "num_steps", "label": "Steps",
                 "min": 1, "max": 200, "default": 100},
                {"type": "slider", "name": "guidance_scale", "label": "CFG Scale",
                 "min": 1.0, "max": 15.0, "default": 4.5, "step": 0.5},
                {"type": "slider", "name": "duration", "label": "Duration (seconds)",
                 "min": 1.0, "max": 30.0, "default": 5.0, "step": 0.5},
                {"type": "dropdown", "name": "sample_rate", "label": "Sample Rate",
                 "options": ["16000", "22050", "44100", "48000"]},
            ],
        }
    
    @classmethod
    def get_available_configs(cls) -> Dict[str, dict]:
        return {
            "default": cls.default_config,
            "audioldm": {
                "type": "model/modular",
                "backbone": {"type": "backbone/unet2d_condition",
                              "pretrained": "cvssp/audioldm-m-full"},
                "codec": {"type": "codec/autoencoder_kl",
                           "pretrained": "cvssp/audioldm-m-full"},
                "conditioner": [{"type": "conditioner/clap"}],
                "guidance": [{"type": "guidance/cfg", "scale": 4.5}],
                "diffusion_process": {"type": "diffusion/process/ddpm"},
            },
        }
