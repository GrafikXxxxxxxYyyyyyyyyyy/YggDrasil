"""VAE для FLUX.2 / FLUX.2 [klein] — AutoencoderKLFlux2.

У Klein (и FLUX.2) свой энкодер/декодер: AutoencoderKLFlux2 из diffusers
(32 канала латентов, 16x даунсэмплинг, batch norm, 2x2 patchify в пайплайне).
Не путать с codec/autoencoder_kl (SD/SDXL).
"""
import torch
from typing import Tuple
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.model.codec import AbstractLatentCodec


def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, C*4, H//2, W//2) для patch_size 2x2."""
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(B, C * 4, H // 2, W // 2)


def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C*4, h, w) -> (B, C, h*2, w*2)."""
    B, C4, h, w = latents.shape
    C = C4 // 4
    latents = latents.reshape(B, C, 2, 2, h, w)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(B, C, h * 2, w * 2)


@register_block("codec/autoencoder_kl_flux2")
class AutoencoderKLFlux2Codec(AbstractLatentCodec):
    """Кодек FLUX.2 / Klein: AutoencoderKLFlux2 (собственный VAE, не AutoencoderKL).

    Конфиг:
        pretrained: str (репо модели, напр. black-forest-labs/FLUX.2-klein-4B)
        subfolder: str (default "vae")
    """

    block_type = "codec/autoencoder_kl_flux2"
    is_trainable = False

    latent_channels = 32
    spatial_scale_factor = 16

    def __init__(self, config: DictConfig):
        super().__init__(config)
        try:
            from diffusers import AutoencoderKLFlux2
        except ImportError:
            raise ImportError("AutoencoderKLFlux2 requires diffusers>=0.37.0.dev0. pip install git+https://github.com/huggingface/diffusers.git")
        load_kwargs = {
            "subfolder": config.get("subfolder", "vae"),
            "torch_dtype": torch.bfloat16 if config.get("bf16", True) else torch.float32,
        }
        if config.get("token") is not None:
            load_kwargs["token"] = config.get("token")
        self.vae = AutoencoderKLFlux2.from_pretrained(
            config.get("pretrained", "black-forest-labs/FLUX.2-klein-4B"),
            **load_kwargs,
        )
        self.vae.requires_grad_(False)
        self.latent_channels = int(getattr(self.vae.config, "latent_channels", 32))
        # vae_scale_factor from block_out_channels (2^(len-1) = 8), *2 for patch -> 16
        self.spatial_scale_factor = 16

    def get_latent_shape(
        self,
        batch_size: int = 1,
        height: int = 512,
        width: int = 512,
    ) -> Tuple[int, ...]:
        return (
            batch_size,
            self.latent_channels,
            height // self.spatial_scale_factor,
            width // self.spatial_scale_factor,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Image [0,1] -> latent (32ch). Для img2img; в txt2img начальные латенты — шум."""
        device = next(self.vae.parameters()).device
        x = x.to(device=device, dtype=self.vae.dtype)
        x = x * 2.0 - 1.0
        with torch.no_grad():
            out = self.vae.encode(x)
            latents = out.latent_dist.mode()
        return latents

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latent (B, 32, H, W) -> image [0,1]. Если пришли packed (B, 128, h, w), делаем unnorm+unpatchify."""
        device = next(self.vae.parameters()).device
        # Поддержка формата после denoise: уже (B, 32, H, W) или packed (B, 128, h, w)
        if z.shape[1] == 128:
            bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(device=z.device, dtype=z.dtype)
            bn_std = torch.sqrt(
                self.vae.bn.running_var.view(1, -1, 1, 1) + getattr(self.vae.config, "batch_norm_eps", 1e-4)
            ).to(device=z.device, dtype=z.dtype)
            z = z * bn_std + bn_mean
            z = _unpatchify_latents(z)
        z = z.to(device=device, dtype=self.vae.dtype)
        with torch.no_grad():
            decoded = self.vae.decode(z, return_dict=False)[0]
        return decoded.float()
