"""DiffusionScheduleBlock — generates timestep schedules.

Turns num_steps into a timestep tensor.
Separates schedule logic from the loop, making it a composable block.

    schedule_block = DiffusionScheduleBlock({"num_train_timesteps": 1000, "schedule_type": "linear"})
    result = schedule_block.process(num_steps=28)
    timesteps = result["timesteps"]  # tensor of shape [28]
"""
from __future__ import annotations

import torch
import math
from typing import Any, Dict
from omegaconf import DictConfig

from yggdrasil.core.block.base import AbstractBlock
from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import InputPort, OutputPort, Port


@register_block("schedule/diffusion")
class DiffusionScheduleBlock(AbstractBlock):
    """Generate timestep schedules for diffusion denoising loops."""
    
    block_type = "schedule/diffusion"
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "schedule/diffusion"}
        super().__init__(config)
        self.num_train_timesteps = int(self.config.get("num_train_timesteps", 1000))
        self.schedule_type = str(self.config.get("schedule_type", "linear"))
    
    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        return {
            "num_steps": InputPort("num_steps", data_type="scalar", optional=True,
                                   description="Number of inference steps"),
            "timesteps": OutputPort("timesteps", data_type="tensor",
                                    description="Timestep schedule tensor"),
        }
    
    def process(self, **port_inputs) -> dict:
        num_steps = port_inputs.get("num_steps", 50)
        if isinstance(num_steps, torch.Tensor):
            num_steps = int(num_steps.item())
        
        if self.schedule_type == "linear":
            timesteps = torch.linspace(
                self.num_train_timesteps - 1, 0, num_steps
            ).long()
        elif self.schedule_type == "quadratic":
            timesteps = torch.tensor([
                int((1 - (i / num_steps) ** 2) * self.num_train_timesteps)
                for i in range(num_steps)
            ]).long()
        elif self.schedule_type == "cosine":
            timesteps = torch.tensor([
                int(math.cos(i / num_steps * math.pi / 2) * self.num_train_timesteps)
                for i in range(num_steps)
            ]).long()
        else:
            timesteps = torch.linspace(
                self.num_train_timesteps - 1, 0, num_steps
            ).long()
        
        return {"timesteps": timesteps, "output": timesteps}
    
    def _forward_impl(self, **kwargs):
        return self.process(**kwargs)
