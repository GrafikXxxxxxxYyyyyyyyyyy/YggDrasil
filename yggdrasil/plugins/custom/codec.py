import torch
from ...core.block.registry import register_block
from ...core.model.codec import AbstractLatentCodec


@register_block("codec/custom")
class CustomCodec(AbstractLatentCodec):
    """Пример кодека (можно сделать Identity, если не нужен VAE)."""
    
    is_trainable = False
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x  # Identity
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z