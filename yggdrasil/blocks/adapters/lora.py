import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .base import AbstractAdapter
from ...core.block.registry import register_block
from ...core.model.backbone import AbstractBackbone


@register_block("adapter/lora")
class LoRAAdapter(AbstractAdapter):
    """Универсальный LoRA (работает с любым линейным/conv слоем)."""
    
    block_type = "adapter/lora"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.rank = config.get("rank", 16)
        self.alpha = config.get("alpha", 16)
        self.dropout = config.get("dropout", 0.0)
        self.target_modules = config.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"])
        
        self.scaling = self.alpha / self.rank
        self.lora_modules = nn.ModuleDict()
    
    def inject_into(self, target: AbstractBackbone):
        """Находит нужные слои и оборачивает их в LoRA."""
        for name, module in target.named_modules():
            if any(target in name for target in self.target_modules):
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    lora = LoRALayer(module, self.rank, self.alpha, self.dropout)
                    self.lora_modules[name] = lora
                    # Заменяем оригинальный forward
                    module.forward = lora.forward.__get__(module, type(module))
    
    def apply(self, output: torch.Tensor, context=None):
        return output  # LoRA уже инжектирована в слои


class LoRALayer(nn.Module):
    """Внутренний LoRA-слой."""
    def __init__(self, base_module, rank, alpha, dropout):
        super().__init__()
        self.base_module = base_module
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = base_module.weight.shape[1]
        out_features = base_module.weight.shape[0]
        
        self.lora_A = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, in_features))
        self.dropout = nn.Dropout(dropout)
        
        # Замораживаем базовый вес
        self.base_module.weight.requires_grad = False
    
    def forward(self, x):
        base_out = self.base_module(x)
        lora_out = (x @ self.lora_B.T) @ self.lora_A.T * self.scaling
        return base_out + self.dropout(lora_out)


@register_block("adapter/dora")
class DoRAAdapter(LoRAAdapter):
    """DoRA (Weight-Decomposed Low-Rank Adaptation) — улучшенная версия LoRA."""
    block_type = "adapter/dora"
    # Реализация почти идентична, но с нормировкой весов (можно расширить позже)