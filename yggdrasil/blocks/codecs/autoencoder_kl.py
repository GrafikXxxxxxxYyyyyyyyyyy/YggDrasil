import torch
from typing import Tuple
from diffusers import AutoencoderKL
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.model.codec import AbstractLatentCodec


@register_block("codec/autoencoder_kl")
class AutoencoderKLCodec(AbstractLatentCodec):
    """VAE для SD 1.5 / SDXL (KL-regularized autoencoder)."""
    
    block_type = "codec/autoencoder_kl"
    is_trainable = False
    
    latent_channels = 4  # SD 1.5 latent space
    spatial_scale_factor = 8  # 512px -> 64 latent

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.vae = AutoencoderKL.from_pretrained(
            config.get("pretrained", "runwayml/stable-diffusion-v1-5"),
            subfolder="vae",
            torch_dtype=torch.float16 if config.get("fp16", True) else torch.float32,
        )
        self.scaling_factor = config.get("scaling_factor", 0.18215)
        self.latent_channels = int(config.get("latent_channels", 4))
        self.spatial_scale_factor = int(config.get("spatial_scale_factor", 8))
    
    def get_latent_shape(
        self,
        batch_size: int = 1,
        height: int = 512,
        width: int = 512,
    ) -> Tuple[int, ...]:
        """Return latent tensor shape for given input dimensions."""
        return (
            batch_size,
            self.latent_channels,
            height // self.spatial_scale_factor,
            width // self.spatial_scale_factor,
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2.0 - 1.0  # [0, 1] -> [-1, 1], VAE expects [-1, 1]
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * self.scaling_factor
        return latents
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z / self.scaling_factor
        # Cast to VAE dtype (latents after solver may be float32 while VAE is float16)
        z = z.to(dtype=self.vae.dtype)
        return self.vae.decode(z).sample