"""Wan 3D Causal VAE — video codec with CausalConv3d.

Encodes/decodes video frames to/from 3D latent space.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.block.port import InputPort, OutputPort, TensorSpec
from ...core.model.codec import AbstractLatentCodec


@register_block("codec/wan_vae")
class WanVideoVAE(AbstractLatentCodec):
    """Wan 3D Causal VAE for video encoding/decoding.
    
    Handles 5D tensors: [B, C, T, H, W]
    """
    
    block_type = "codec/wan_vae"
    is_trainable = False
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "codec/wan_vae"}
        super().__init__(config)
        self._model = None
        self.pretrained = self.config.get("pretrained", "Wan-AI/Wan2.1-T2V-14B-Diffusers")
        self.latent_channels = int(self.config.get("latent_channels", 16))
        self.scaling_factor = float(self.config.get("scaling_factor", 0.18215))
        self._build_model()
    
    @classmethod
    def declare_io(cls) -> dict:
        return {
            "pixel_data": InputPort("pixel_data", spec=TensorSpec(space="pixel"), optional=True,
                                    description="Video frames [B, C, T, H, W]"),
            "latent": InputPort("latent", spec=TensorSpec(space="latent"), optional=True,
                                description="Video latent [B, C_lat, T, H, W]"),
            "operation": InputPort("operation", data_type="scalar", optional=True),
            "encoded": OutputPort("encoded", spec=TensorSpec(space="latent")),
            "decoded": OutputPort("decoded", spec=TensorSpec(space="pixel")),
        }
    
    def _build_model(self):
        try:
            from diffusers import AutoencoderKLWan
            self._model = AutoencoderKLWan.from_pretrained(
                self.pretrained, subfolder="vae",
                torch_dtype=torch.float32,  # VAE always float32
            )
            self._model.requires_grad_(False)
        except Exception:
            # Stub: identity codec for testing
            self._model = None
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self._model is not None:
            with torch.no_grad():
                latent = self._model.encode(x).latent_dist.sample()
                return latent * self.scaling_factor
        # Stub: downsample spatially
        if x.dim() == 5:
            return torch.nn.functional.avg_pool3d(x, kernel_size=(1, 8, 8))
        return torch.nn.functional.avg_pool2d(x, kernel_size=8)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self._model is not None:
            with torch.no_grad():
                z = z / self.scaling_factor
                return self._model.decode(z).sample
        # Stub: upsample spatially
        if z.dim() == 5:
            return torch.nn.functional.interpolate(
                z.view(-1, z.shape[2], z.shape[3], z.shape[4]),
                scale_factor=8, mode="nearest",
            ).view(z.shape[0], z.shape[1], z.shape[2], z.shape[3] * 8, z.shape[4] * 8)
        return torch.nn.functional.interpolate(z, scale_factor=8, mode="nearest")
