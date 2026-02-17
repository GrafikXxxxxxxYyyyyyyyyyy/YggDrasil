from __future__ import annotations

import torch
from abc import abstractmethod
from typing import Dict, Any, Optional
from omegaconf import DictConfig

from ...core.block.base import AbstractBaseBlock
from ...core.block.registry import register_block
from ...core.block.port import Port, InputPort, OutputPort, TensorSpec


@register_block("diffusion/process/abstract")
class AbstractDiffusionProcess(AbstractBaseBlock):
    """Abstract diffusion process (DDPM, Flow Matching, etc.) — no slots; schedule from graph if needed."""

    block_type = "diffusion/process/abstract"

    @classmethod
    def declare_io(cls) -> dict:
        return {
            "x0": InputPort("x0", spec=TensorSpec(space="latent"), description="Clean data"),
            "t": InputPort("t", data_type="tensor", description="Timestep"),
            "noise": InputPort("noise", optional=True, description="Optional noise"),
            "xt": OutputPort("xt", spec=TensorSpec(space="latent"), description="Noised data"),
            "target": OutputPort("target", description="Target for loss computation"),
        }

    def process(self, **port_inputs) -> dict:
        x0 = port_inputs.get("x0")
        t = port_inputs.get("t")
        noise = port_inputs.get("noise")
        result = self.forward_process(x0, t, noise)
        return result

    def _forward_impl(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Требуется AbstractBaseBlock; делегирует в forward_process."""
        return self.forward_process(x0, t, noise)

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