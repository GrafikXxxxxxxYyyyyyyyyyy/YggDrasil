from __future__ import annotations

import torch
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class DiffusionTensor:
    """Умный тензор для любой модальности.
    
    Хранит:
    - raw tensor (любая размерность)
    - metadata (модальность, scale, mean, original_shape и т.д.)
    - удобные методы to/from модальности
    """
    
    tensor: torch.Tensor
    modality: str = "unknown"                    # "image", "audio", "timeseries", "molecular"...
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not isinstance(self.tensor, torch.Tensor):
            self.tensor = torch.as_tensor(self.tensor)
    
    def to(self, device: torch.device | str) -> DiffusionTensor:
        return DiffusionTensor(
            tensor=self.tensor.to(device),
            modality=self.modality,
            metadata=self.metadata.copy()
        )
    
    def cpu(self) -> DiffusionTensor:
        return self.to("cpu")
    
    def detach(self) -> DiffusionTensor:
        return DiffusionTensor(
            tensor=self.tensor.detach(),
            modality=self.modality,
            metadata=self.metadata.copy()
        )
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.tensor.shape
    
    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype
    
    def __repr__(self):
        return f"<DiffusionTensor {self.modality} shape={self.shape} device={self.tensor.device}>"
    
    # ==================== Конвертеры ====================
    
    def to_image(self) -> torch.Tensor:
        """Для модальностей, которые могут стать изображением."""
        if self.modality == "image":
            return self.tensor
        raise ValueError(f"Cannot convert {self.modality} to image")
    
    def to_audio(self) -> torch.Tensor:
        if self.modality == "audio":
            return self.tensor
        raise ValueError(f"Cannot convert {self.modality} to audio")
    
    @classmethod
    def from_modality(cls, data: Any, modality: str, **metadata) -> DiffusionTensor:
        """Универсальный конструктор из данных модальности."""
        if modality == "image":
            tensor = torch.as_tensor(data).float() / 127.5 - 1.0
        elif modality == "audio":
            tensor = torch.as_tensor(data).float()
        else:
            tensor = torch.as_tensor(data)
        
        return cls(tensor=tensor, modality=modality, metadata=metadata)