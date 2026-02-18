from __future__ import annotations

import torch
from abc import abstractmethod
from typing import Dict, Any, Optional, Callable

from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from ...core.block.base import AbstractBaseBlock
from ...core.block.registry import register_block
from ...core.block.builder import BlockBuilder
from .sampler import DiffusionSampler
from .state import DiffusionState
from ...core.model.modular import ModularDiffusionModel


@register_block("engine/pipeline/abstract")
class AbstractPipeline(AbstractBaseBlock):
    """Base pipeline (train / infer) — graph engine; model and sampler from config."""

    block_type = "engine/pipeline/abstract"

    def __init__(self, config):
        super().__init__(config)
        model_cfg = self.config.get("model")
        self._model: ModularDiffusionModel = (
            model_cfg if isinstance(model_cfg, ModularDiffusionModel)
            else BlockBuilder.build(model_cfg) if isinstance(model_cfg, dict) else None
        )
        sampler_cfg = self.config.get("sampler", {"type": "engine/sampler"})
        self._sampler: Optional[DiffusionSampler] = (
            sampler_cfg if isinstance(sampler_cfg, DiffusionSampler)
            else BlockBuilder.build(sampler_cfg) if isinstance(sampler_cfg, dict) else None
        )
        if self._sampler is not None and self._model is not None:
            self._sampler._model = self._model

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Один шаг обучения."""
        pass
    
    @abstractmethod
    def infer_step(self, condition: Dict[str, Any], **kwargs) -> torch.Tensor:
        """Один шаг инференса (полная генерация)."""
        pass
    
    def save(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config, path / "pipeline_config.yaml")
        if self._model is not None:
            self._model.save(path / "model")
        if self._sampler is not None:
            self._sampler.save(path / "sampler")

    @classmethod
    def load(cls, path: Path | str) -> AbstractPipeline:
        path = Path(path)
        config = OmegaConf.load(path / "pipeline_config.yaml")
        instance = cls(config)
        instance._model = ModularDiffusionModel.load(path / "model")
        if instance._sampler is not None:
            instance._sampler._model = instance._model
        return instance