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
        load_kwargs = {
            "subfolder": config.get("subfolder", "vae"),
            "torch_dtype": torch.float16 if config.get("fp16", True) else torch.float32,
        }
        if config.get("token") is not None:
            load_kwargs["token"] = config.get("token")
        self.vae = AutoencoderKL.from_pretrained(
            config.get("pretrained", "runwayml/stable-diffusion-v1-5"),
            **load_kwargs,
        )
        self.vae.requires_grad_(False)
        # Prefer config; else VAE's config (SD3 uses 1.5305 + shift 0.0609; SD 1.5 default 0.18215)
        scaling = config.get("scaling_factor")
        if scaling is None and getattr(self.vae, "config", None) is not None:
            scaling = getattr(self.vae.config, "scaling_factor", None)
        self.scaling_factor = float(scaling if scaling is not None else 0.18215)
        shift = config.get("shift_factor")
        if shift is None and getattr(self.vae, "config", None) is not None:
            shift = getattr(self.vae.config, "shift_factor", None)
        self.shift_factor = float(shift) if shift is not None else 0.0
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
            sample = self.vae.encode(x).latent_dist.sample()
        # Stored latent: (vae_sample - shift) * scale (Diffusers SD3 convention)
        latents = (sample - self.shift_factor) * self.scaling_factor
        return latents
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Video: (B, C, T, h, w) -> decode frame-by-frame
        if z.dim() == 5:
            B, C, T, h, w = z.shape
            z_2d = z.permute(0, 2, 1, 3, 4).reshape(B * T, C, h, w)
            out_2d = self.decode(z_2d)
            return out_2d.reshape(B, T, *out_2d.shape[1:]).permute(0, 2, 1, 3, 4)
        # Inverse of stored latent -> VAE input (Diffusers: latents/scale + shift)
        z = (z / self.scaling_factor) + self.shift_factor
        vae_device = next(self.vae.parameters()).device

        # SDXL VAE in float16 produces NaN (overflow); use float32 for decode.
        # SD3 (scale 1.53): float32 for correct decode; MPS: float32 for stability.
        use_fp32_decode = (
            vae_device.type == "mps"
            or self.scaling_factor == 0.13025  # SDXL
            or self.scaling_factor >= 1.0  # SD3 (1.53), Flux — correct range, less artifact
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