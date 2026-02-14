# yggdrasil/training/loss.py
"""Универсальные loss-функции для обучения диффузионных моделей любой модальности.

Каждый loss — это Lego-кирпичик, который можно использовать с любой моделью.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from omegaconf import DictConfig

from ..core.block.base import AbstractBlock
from ..core.block.registry import register_block


@register_block("training/loss/abstract")
class DiffusionLoss(AbstractBlock, ABC):
    """Базовый loss для обучения диффузии.
    
    Поддерживает любую параметризацию: epsilon, v-prediction, flow matching, score.
    """
    
    block_type = "training/loss/abstract"
    
    def _define_slots(self):
        return {}
    
    @abstractmethod
    def compute(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        timestep: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Вычисление loss.
        
        Returns:
            {"loss": total_loss, ...optional_components}
        """
        pass
    
    def _forward_impl(self, model_output, target, timestep, **kwargs):
        return self.compute(model_output, target, timestep, **kwargs)


@register_block("training/loss/epsilon")
class EpsilonLoss(DiffusionLoss):
    """MSE loss для epsilon-параметризации (SD 1.5, SDXL)."""
    
    block_type = "training/loss/epsilon"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {})
        self.loss_type = self.config.get("loss_type", "mse")  # mse, l1, huber
    
    def compute(self, model_output, target, timestep, weights=None, **kwargs):
        if self.loss_type == "mse":
            loss = F.mse_loss(model_output, target, reduction="none")
        elif self.loss_type == "l1":
            loss = F.l1_loss(model_output, target, reduction="none")
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(model_output, target, reduction="none")
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        # Взвешивание по таймстепам (SNR weighting и т.д.)
        if weights is not None:
            loss = loss * weights.view(-1, *([1] * (loss.ndim - 1)))
        
        return {"loss": loss.mean(), "loss_per_sample": loss.mean(dim=list(range(1, loss.ndim)))}


@register_block("training/loss/velocity")
class VelocityLoss(DiffusionLoss):
    """Loss для v-prediction параметризации (SD 2.x)."""
    
    block_type = "training/loss/velocity"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {})
    
    def compute(self, model_output, target, timestep, weights=None, **kwargs):
        loss = F.mse_loss(model_output, target, reduction="none")
        if weights is not None:
            loss = loss * weights.view(-1, *([1] * (loss.ndim - 1)))
        return {"loss": loss.mean(), "loss_per_sample": loss.mean(dim=list(range(1, loss.ndim)))}


@register_block("training/loss/flow_matching")
class FlowMatchingLoss(DiffusionLoss):
    """Loss для Flow Matching (Flux, SD3, Stable Audio)."""
    
    block_type = "training/loss/flow_matching"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {})
        self.sigma_min = float(self.config.get("sigma_min", 1e-5))
    
    def compute(self, model_output, target, timestep, weights=None, **kwargs):
        # target = velocity = x1 - x0
        loss = F.mse_loss(model_output, target, reduction="none")
        
        # Опциональное time-dependent weighting
        if weights is not None:
            loss = loss * weights.view(-1, *([1] * (loss.ndim - 1)))
        
        return {"loss": loss.mean(), "loss_per_sample": loss.mean(dim=list(range(1, loss.ndim)))}


@register_block("training/loss/score")
class ScoreLoss(DiffusionLoss):
    """Loss для score-based моделей (NCSN, SDE-diffusion)."""
    
    block_type = "training/loss/score"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {})
    
    def compute(self, model_output, target, timestep, weights=None, **kwargs):
        loss = F.mse_loss(model_output, target, reduction="none")
        # Score weighting: lambda(t) = sigma(t)^2
        sigma = kwargs.get("sigma", None)
        if sigma is not None:
            loss = loss * (sigma ** 2).view(-1, *([1] * (loss.ndim - 1)))
        if weights is not None:
            loss = loss * weights.view(-1, *([1] * (loss.ndim - 1)))
        return {"loss": loss.mean(), "loss_per_sample": loss.mean(dim=list(range(1, loss.ndim)))}


@register_block("training/loss/composite")
class CompositeLoss(DiffusionLoss):
    """Комбинированный loss из нескольких компонентов.
    
    Пример конфига:
        type: training/loss/composite
        components:
          - type: training/loss/epsilon
            weight: 1.0
          - type: training/loss/perceptual
            weight: 0.1
    """
    
    block_type = "training/loss/composite"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {})
        from ..core.block.builder import BlockBuilder
        self.components = []
        self.component_weights = []
        for comp_cfg in self.config.get("components", []):
            weight = comp_cfg.get("weight", 1.0)
            self.components.append(BlockBuilder.build(comp_cfg))
            self.component_weights.append(weight)
    
    def compute(self, model_output, target, timestep, weights=None, **kwargs):
        total_loss = torch.tensor(0.0, device=model_output.device)
        results = {}
        for i, (comp, w) in enumerate(zip(self.components, self.component_weights)):
            comp_result = comp.compute(model_output, target, timestep, weights, **kwargs)
            total_loss = total_loss + w * comp_result["loss"]
            results[f"component_{i}_loss"] = comp_result["loss"]
        results["loss"] = total_loss
        return results
