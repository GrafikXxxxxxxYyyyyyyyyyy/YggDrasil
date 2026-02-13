import torch
from diffusers import AutoencoderKL
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.model.codec import AbstractLatentCodec


@register_block("codec/autoencoder_kl")
class AutoencoderKLCodec(AbstractLatentCodec):
    """VAE для SD 1.5 (scaling factor = 0.18215)."""
    
    block_type = "codec/autoencoder_kl"
    is_trainable = False
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.vae = AutoencoderKL.from_pretrained(
            config.get("pretrained", "runwayml/stable-diffusion-v1-5"),
            subfolder="vae",
            torch_dtype=torch.float16 if config.get("fp16", True) else torch.float32,
        )
        self.scaling_factor = config.get("scaling_factor", 0.18215)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2.0 - 1.0  # [-1, 1] → [0, 1] → VAE ожидает [-1, 1]
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * self.scaling_factor
        return latents
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z / self.scaling_factor
        return self.vae.decode(z).sample