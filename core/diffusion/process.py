from __future__ import annotations

import torch
from abc import abstractmethod
from typing import Dict, Any, Optional
from omegaconf import DictConfig

from ...core.block.base import AbstractBlock
from ...core.block.registry import register_block
from ...core.block.slot import Slot

from .noise.schedule import NoiseSchedule


@register_block("diffusion/process/abstract")
class AbstractDiffusionProcess(AbstractBlock):
    """Абстрактный диффузионный процесс (DDPM, Flow Matching, Consistency, SDE и т.д.).
    
    Это главный математический Lego-кирпичик.
    """
    
    block_type = "diffusion/process/abstract"
    
    def _define_slots(self) -> Dict[str, Slot]:
        return {
            "noise_schedule": Slot(
                name="noise_schedule",
                accepts=NoiseSchedule,
                multiple=False,
                optional=True
            )
        }
    
    @abstractmethod
    def forward_process(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """x0 + noise → xt (forward diffusion)."""
        pass
    
    @abstractmethod
    def reverse_step(
        self,
        model_output: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Один шаг обратного процесса (используется в sampler.step)."""
        pass
    
    @abstractmethod
    def predict_x0(
        self,
        model_output: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Предсказание x0 из model_output (noise_pred / velocity / etc)."""
        pass
    
    @abstractmethod
    def predict_velocity(
        self,
        model_output: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Для flow-matching моделей."""
        pass