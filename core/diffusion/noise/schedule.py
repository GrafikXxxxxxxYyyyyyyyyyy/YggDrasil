import torch
from abc import abstractmethod
from typing import Optional
from omegaconf import DictConfig

from ....core.block.base import AbstractBlock
from ....core.block.registry import register_block


@register_block("noise/schedule/abstract")
class NoiseSchedule(AbstractBlock):
    """Расписание шума (cosine, linear, sigmoid, custom)."""
    
    block_type = "noise/schedule/abstract"
    
    def _define_slots(self):
        return {}
    
    @abstractmethod
    def get_timesteps(self, num_steps: int) -> torch.Tensor:
        """Возвращает tensor таймстепов [0.0 ... 1.0]"""
        pass
    
    @abstractmethod
    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """α(t) для DDPM-style"""
        pass
    
    @abstractmethod
    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """σ(t) для variance"""
        pass