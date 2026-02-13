from omegaconf import DictConfig
import torch

from .base import AbstractAdapter
from ...core.block.registry import register_block


@register_block("adapter/ip_adapter")
class IPAdapter(AbstractAdapter):
    """IP-Adapter — добавляет image prompt через cross-attention."""
    
    block_type = "adapter/ip_adapter"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.scale = config.get("scale", 1.0)
        # В реальной реализации — загружается из HF или кастомный projector
    
    def inject_into(self, target):
        # Инжектирует дополнительный cross-attention в каждый attention блок
        for name, module in target.named_modules():
            if "attn2" in name and hasattr(module, "to_k"):
                # Добавляем projector и т.д. (упрощённо)
                pass
    
    def apply(self, output: torch.Tensor, context=None):
        # Логика IP-Adapter обычно в conditioner
        return output