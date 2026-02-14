import torch
from typing import Tuple
from diffusers import AutoencoderKL
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.model.codec import AbstractLatentCodec


@register_block("codec/autoencoder_kl")
class AutoencoderKLCodec(AbstractLatentCodec):
    """VAE для SD 1.5 / SDXL (KL-regularized autoencoder).
    
    Загружается в float16 для экономии памяти.
    Декодирование на MPS выполняется в float32 для стабильности
    (GroupNorm/upsample в float16 на MPS может давать NaN).
    """
    
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
        self.vae.requires_grad_(False)
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
        vae_device = next(self.vae.parameters()).device
        x = x.to(device=vae_device, dtype=self.vae.dtype)
        x = x * 2.0 - 1.0  # [0, 1] -> [-1, 1], VAE expects [-1, 1]
        with torch.no_grad():
            latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * self.scaling_factor
        return latents
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z / self.scaling_factor
        vae_device = next(self.vae.parameters()).device

        # SDXL VAE in float16 produces NaN (overflow); use float32 for decode.
        # MPS: float32 for stability (GroupNorm/conv in fp16 can artifact).
        use_fp32_decode = (
            vae_device.type == "mps"
            or self.scaling_factor == 0.13025  # SDXL
        )
        if use_fp32_decode:
            z = z.to(device=vae_device, dtype=torch.float32)
            with torch.no_grad():
                self.vae.to(dtype=torch.float32)
                decoded = self.vae.decode(z).sample
        else:
            z = z.to(device=vae_device, dtype=self.vae.dtype)
            with torch.no_grad():
                decoded = self.vae.decode(z).sample
        # Return float32 for stable postprocess (denorm to [0,1])
        return decoded.float()