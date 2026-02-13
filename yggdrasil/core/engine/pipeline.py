from __future__ import annotations

import torch
from abc import abstractmethod
from typing import Dict, Any, Optional, Callable
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from ...core.block.base import AbstractBlock
from ...core.block.registry import register_block
from ...core.block.slot import Slot

from .sampler import DiffusionSampler
from .state import DiffusionState
from ...core.model.modular import ModularDiffusionModel


@register_block("engine/pipeline/abstract")
class AbstractPipeline(AbstractBlock):
    """Базовый пайплайн (train / infer / distillation / multi-modal).
    
    Один класс — все режимы. Наследники реализуют конкретные сценарии.
    """
    
    block_type = "engine/pipeline/abstract"
    
    def _define_slots(self) -> Dict[str, Slot]:
        return {
            "model": Slot(
                name="model",
                accepts=ModularDiffusionModel,
                multiple=False,
                optional=False
            ),
            "sampler": Slot(
                name="sampler",
                accepts=DiffusionSampler,
                multiple=False,
                optional=True,
                default={"type": "engine/sampler"}
            ),
        }
    
    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Один шаг обучения."""
        pass
    
    @abstractmethod
    def infer_step(self, condition: Dict[str, Any], **kwargs) -> torch.Tensor:
        """Один шаг инференса (полная генерация)."""
        pass
    
    def save(self, path: Path | str):
        """Сохранение всего пайплайна."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config, path / "pipeline_config.yaml")
        self._slot_children["model"].save(path / "model")
        if "sampler" in self._slot_children:
            self._slot_children["sampler"].save(path / "sampler")
    
    @classmethod
    def load(cls, path: Path | str) -> AbstractPipeline:
        path = Path(path)
        config = OmegaConf.load(path / "pipeline_config.yaml")
        instance = cls(config)
        instance._slot_children["model"] = ModularDiffusionModel.load(path / "model")
        return instance