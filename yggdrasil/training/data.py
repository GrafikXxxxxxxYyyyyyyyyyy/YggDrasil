# yggdrasil/training/data.py
"""Абстракции для источников данных — любая модальность.

Данные проходят через:
    DataSource → DataLoader → Trainer
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List, Tuple
from pathlib import Path
from omegaconf import DictConfig


class AbstractDataSource(Dataset, ABC):
    """Абстрактный источник данных для обучения.
    
    Каждая модальность (изображения, аудио, молекулы, ...) реализует свой наследник.
    Должен возвращать dict: {"data": Tensor, "condition": {...}, ...}
    """
    
    @abstractmethod
    def __len__(self) -> int:
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Возвращает один пример.
        
        Минимальный контракт:
            {"data": Tensor}  — данные в исходном пространстве
            
        Опционально:
            {"data": Tensor, "condition": {"text": str, "class": int, ...}}
        """
        pass
    
    def get_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        **kwargs
    ) -> DataLoader:
        """Создать DataLoader из этого источника."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn,
            **kwargs
        )
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Сборка batch из списка примеров. Можно переопределить."""
        result = {}
        keys = batch[0].keys()
        for key in keys:
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            elif isinstance(values[0], dict):
                result[key] = {
                    k: [v[k] for v in values] if not isinstance(values[0][k], torch.Tensor)
                    else torch.stack([v[k] for v in values])
                    for k in values[0].keys()
                }
            else:
                result[key] = values
        return result


class ImageFolderSource(AbstractDataSource):
    """Источник данных: папка с изображениями.
    
    Поддерживает:
    - Без условий (unconditional)
    - С текстовым условием (из .txt файлов рядом с изображениями)
    - С классами (из подпапок)
    """
    
    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    
    def __init__(
        self,
        root: str | Path,
        resolution: int = 512,
        center_crop: bool = True,
        flip_p: float = 0.5,
        caption_ext: str = ".txt",
    ):
        self.root = Path(root)
        self.resolution = resolution
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.caption_ext = caption_ext
        
        # Собираем все изображения
        self.image_paths = sorted([
            p for p in self.root.rglob("*")
            if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ])
        
        if len(self.image_paths) == 0:
            raise ValueError(f"Нет изображений в {self.root}")
        
        # Трансформации
        self._transform = self._build_transform()
    
    def _build_transform(self):
        """Ленивый импорт torchvision."""
        try:
            from torchvision import transforms
            t = [transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.LANCZOS)]
            if self.center_crop:
                t.append(transforms.CenterCrop(self.resolution))
            else:
                t.append(transforms.RandomCrop(self.resolution))
            if self.flip_p > 0:
                t.append(transforms.RandomHorizontalFlip(self.flip_p))
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize([0.5], [0.5]))  # [-1, 1]
            return transforms.Compose(t)
        except ImportError:
            return None
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        from PIL import Image
        
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self._transform is not None:
            image = self._transform(image)
        else:
            # Fallback без torchvision
            import numpy as np
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0
        
        result = {"data": image}
        
        # Текстовое условие (из .txt файла)
        caption_path = img_path.with_suffix(self.caption_ext)
        if caption_path.exists():
            result["condition"] = {"text": caption_path.read_text().strip()}
        
        # Класс из имени подпапки
        if img_path.parent != self.root:
            result.setdefault("condition", {})["class_name"] = img_path.parent.name
        
        return result


class TensorDataSource(AbstractDataSource):
    """Источник данных из готовых тензоров (для любой модальности).
    
    Идеально для:
    - Временных рядов
    - Молекулярных данных  
    - Предобработанных латентов
    - Кастомных модальностей
    """
    
    def __init__(
        self,
        data: torch.Tensor,
        conditions: Optional[Dict[str, Any]] = None,
    ):
        self.data = data
        self.conditions = conditions
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        result = {"data": self.data[idx]}
        if self.conditions is not None:
            result["condition"] = {
                k: v[idx] if isinstance(v, (torch.Tensor, list)) else v
                for k, v in self.conditions.items()
            }
        return result


class LatentCacheSource(AbstractDataSource):
    """Источник данных из предварительно закешированных латентов.
    
    Экономит VRAM — не нужно держать VAE encoder при обучении.
    """
    
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("*.pt"))
        if not self.files:
            raise ValueError(f"Нет кешированных латентов в {self.cache_dir}")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return torch.load(self.files[idx], weights_only=False)
    
    @staticmethod
    def cache_dataset(
        data_source: AbstractDataSource,
        model: "ModularDiffusionModel",
        output_dir: str | Path,
        batch_size: int = 4,
        device: str = "cuda",
    ):
        """Кешировать латенты из dataset через codec модели."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        loader = data_source.get_dataloader(batch_size=batch_size, shuffle=False)
        idx = 0
        
        with torch.no_grad():
            for batch in loader:
                data = batch["data"].to(device)
                latents = model.encode(data)
                
                for i in range(latents.shape[0]):
                    item = {"data": latents[i].cpu()}
                    if "condition" in batch:
                        cond = {}
                        for k, v in batch["condition"].items():
                            cond[k] = v[i] if isinstance(v, (torch.Tensor, list)) else v
                        item["condition"] = cond
                    torch.save(item, output_dir / f"{idx:06d}.pt")
                    idx += 1
        
        print(f"Закешировано {idx} латентов в {output_dir}")
