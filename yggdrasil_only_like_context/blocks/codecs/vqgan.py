"""VQGAN (Vector-Quantized GAN) codec.

Quantizes latents to a discrete codebook. Used by some image/audio models.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.model.codec import AbstractLatentCodec


@register_block("codec/vqgan")
class VQGANCodec(AbstractLatentCodec):
    """VQGAN codec with vector quantization.
    
    Wraps a VQ-VAE/VQGAN model. For now provides a structural placeholder;
    real-world usage would load pre-trained weights from taming-transformers
    or similar.
    """
    
    block_type = "codec/vqgan"
    is_trainable = False
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {})
        self.latent_channels = int(self.config.get("latent_channels", 256))
        self.codebook_size = int(self.config.get("codebook_size", 8192))
        self.spatial_scale_factor = int(self.config.get("spatial_scale_factor", 16))
        
        in_channels = int(self.config.get("in_channels", 3))
        hidden_dim = int(self.config.get("hidden_dim", 128))
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 4, self.latent_channels, 4, 2, 1),
        )
        
        # Codebook
        self.codebook = nn.Embedding(self.codebook_size, self.latent_channels)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent_channels, hidden_dim * 4, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, in_channels, 4, 2, 1),
        )
    
    def get_latent_shape(
        self,
        batch_size: int = 1,
        height: int = 256,
        width: int = 256,
    ) -> Tuple[int, ...]:
        return (
            batch_size,
            self.latent_channels,
            height // self.spatial_scale_factor,
            width // self.spatial_scale_factor,
        )
    
    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize continuous latents to codebook entries."""
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Nearest codebook entry
        dists = torch.cdist(z_flat, self.codebook.weight)
        indices = dists.argmin(dim=-1)
        z_q = self.codebook(indices).reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        return z_q, indices
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z_q, _ = self.quantize(z)
        return z_q
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
