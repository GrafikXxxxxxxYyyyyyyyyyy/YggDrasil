from __future__ import annotations

import torch
from omegaconf import DictConfig

from ...plugins.base import AbstractPlugin
from ...core.block.registry import register_block
from ...core.utils.tensor import DiffusionTensor


@register_block("plugin/custom")
class CustomModality(AbstractPlugin):
    """Шаблон новой модальности.
    
    Замени "Custom" на название своей модальности (TimeseriesModality, MidiModality и т.д.)
    """
    
    name = "custom"
    default_config = "plugins/custom/config.yaml"
    
    @classmethod
    def register_blocks(cls):
        """Регистрируем все кирпичики этой модальности."""
        # Регистрируем backbone, codec и т.д.
        from .backbone import CustomBackbone
        from .codec import CustomCodec
        
        # Можно регистрировать через @register_block в самих файлах,
        # но здесь удобно собрать всё в одном месте
    
    # ==================== ИНТЕРФЕЙС МОДАЛЬНОСТИ ====================
    
    @staticmethod
    def to_tensor(data: Any) -> DiffusionTensor:
        """Преобразование сырых данных в DiffusionTensor."""
        # Пример для временных рядов:
        # tensor = torch.tensor(data, dtype=torch.float32)
        tensor = torch.as_tensor(data).float()
        return DiffusionTensor(tensor=tensor, modality="custom")
    
    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> Any:
        """Обратно в формат пользователя."""
        return tensor.cpu().numpy()
    
    @staticmethod
    def visualize(tensor: torch.Tensor, **kwargs):
        """Визуализация (matplotlib, plotly, wandb и т.д.)."""
        import matplotlib.pyplot as plt
        plt.plot(tensor[0].cpu().numpy())
        plt.title("Custom Modality Visualization")
        plt.show()
    
    @staticmethod
    def get_default_pipeline() -> dict:
        """Дефолтная сборка для этой модальности."""
        return {
            "type": "model/modular",
            "backbone": {"type": "backbone/custom"},
            "codec": {"type": "codec/custom"},
            "conditioner": {"type": "conditioner/text_clip"},
            "diffusion_process": {"type": "diffusion/process/flow/rectified"}
        }