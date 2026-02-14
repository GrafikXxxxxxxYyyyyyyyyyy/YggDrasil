"""Identity codec -- pass-through for pixel-space or raw-space diffusion."""
import torch
from typing import Tuple
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.model.codec import AbstractLatentCodec


@register_block("codec/identity")
class IdentityCodec(AbstractLatentCodec):
    """Pass-through codec for diffusion directly in data space.
    
    Used when no latent encoding is needed (pixel-space diffusion,
    molecular coordinates, raw time series, etc.).
    """
    
    block_type = "codec/identity"
    is_trainable = False
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {})
        self.latent_channels = int(self.config.get("latent_channels", 3))
        self.spatial_scale_factor = 1
    
    def get_latent_shape(
        self,
        batch_size: int = 1,
        height: int = 256,
        width: int = 256,
    ) -> Tuple[int, ...]:
        return (batch_size, self.latent_channels, height, width)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z
