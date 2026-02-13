from __future__ import annotations

import torch
from abc import abstractmethod
from typing import Tuple, Optional
from omegaconf import DictConfig

from ...core.block.base import AbstractBlock
from ...core.block.registry import register_block


@register_block("codec/abstract")
class AbstractLatentCodec(AbstractBlock):
    """Абстрактный кодек (VAE, VQGAN, Encodec, Identity, GaussianSplatting...)."""
    
    block_type = "codec/abstract"
    is_trainable: bool = False          # По умолчанию заморожен (как VAE)
    
    def _define_slots(self):
        return {}  # Кодеки редко имеют слоты
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x → latents (обычно [B, C_latent, ...])"""
        pass
    
    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """latents → reconstructed data"""
        pass
    
    def _forward_impl(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Требуется AbstractBlock: делегирует в forward."""
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Для обучения: вернуть latents + kl_loss (если есть)."""
        z = self.encode(x)
        return z, None