from __future__ import annotations

import torch
from abc import abstractmethod
from typing import Tuple, Optional
from omegaconf import DictConfig

from ...core.block.base import AbstractBlock
from ...core.block.registry import register_block
from ...core.block.port import Port, InputPort, OutputPort, TensorSpec


@register_block("codec/abstract")
class AbstractLatentCodec(AbstractBlock):
    """Абстрактный кодек (VAE, VQGAN, Encodec, Identity, GaussianSplatting...)."""
    
    block_type = "codec/abstract"
    is_trainable: bool = False          # По умолчанию заморожен (как VAE)
    
    @classmethod
    def declare_io(cls) -> dict:
        return {
            "pixel_data": InputPort("pixel_data", spec=TensorSpec(space="pixel"), optional=True, description="Input data (pixel space)"),
            "latent": InputPort("latent", spec=TensorSpec(space="latent"), optional=True, description="Latent for decoding"),
            "operation": InputPort("operation", data_type="scalar", optional=True, description="'encode' or 'decode'"),
            "encoded": OutputPort("encoded", spec=TensorSpec(space="latent"), description="Encoded latent"),
            "decoded": OutputPort("decoded", spec=TensorSpec(space="pixel"), description="Decoded output"),
        }
    
    def process(self, **port_inputs) -> dict:
        operation = port_inputs.get("operation", "encode")
        if operation == "decode" or ("latent" in port_inputs and "pixel_data" not in port_inputs):
            z = port_inputs.get("latent", port_inputs.get("pixel_data"))
            decoded = self.decode(z)
            return {"decoded": decoded, "output": decoded}
        else:
            x = port_inputs.get("pixel_data", port_inputs.get("latent"))
            encoded = self.encode(x)
            return {"encoded": encoded, "output": encoded}
    
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