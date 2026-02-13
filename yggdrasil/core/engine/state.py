from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from omegaconf import DictConfig

from ...core.block.base import AbstractBlock
from ...core.block.registry import register_block


@register_block("engine/state")
@dataclass
class DiffusionState(AbstractBlock):
    """Состояние диффузионного процесса на каждом шаге.
    
    Это Lego-кирпичик, который передаётся между sampler, loop и callbacks.
    """
    
    block_type = "engine/state"
    
    # Основные поля
    latents: torch.Tensor
    timestep: torch.Tensor
    condition: Dict[str, Any] = field(default_factory=dict)
    
    # Дополнительные кэши
    prev_latents: Optional[torch.Tensor] = None
    noise: Optional[torch.Tensor] = None
    model_output: Optional[torch.Tensor] = None
    
    # Метаданные (для модальностей)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # AbstractBlock.__init__ не нужен, т.к. это dataclass
        pass
    
    def to(self, device: torch.device) -> DiffusionState:
        """Перемещение на устройство."""
        self.latents = self.latents.to(device)
        self.timestep = self.timestep.to(device)
        if self.prev_latents is not None:
            self.prev_latents = self.prev_latents.to(device)
        if self.noise is not None:
            self.noise = self.noise.to(device)
        return self
    
    def clone(self) -> DiffusionState:
        """Глубокая копия для safety."""
        return DiffusionState(
            latents=self.latents.clone(),
            timestep=self.timestep.clone(),
            condition=self.condition.copy(),
            prev_latents=self.prev_latents.clone() if self.prev_latents is not None else None,
            noise=self.noise.clone() if self.noise is not None else None,
            metadata=self.metadata.copy()
        )