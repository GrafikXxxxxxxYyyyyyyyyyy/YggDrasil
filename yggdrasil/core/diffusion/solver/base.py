import torch
from abc import abstractmethod
from typing import Any, Dict

from ....core.block.base import AbstractBlock
from ....core.block.registry import register_block
from ..process import AbstractDiffusionProcess


@register_block("diffusion/solver/abstract")
class AbstractSolver(AbstractBlock):
    """Абстрактный солвер (DDIM, Heun, Euler, DPM и т.д.)."""
    
    block_type = "diffusion/solver/abstract"
    
    def _define_slots(self):
        return {}
    
    @abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        current_latents: torch.Tensor,
        timestep: torch.Tensor,
        process: AbstractDiffusionProcess,
        **kwargs: Any
    ) -> torch.Tensor:
        """Один шаг солвера."""
        pass