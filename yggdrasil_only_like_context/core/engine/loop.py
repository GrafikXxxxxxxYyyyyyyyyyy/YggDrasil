from __future__ import annotations

import torch
from typing import Callable, List, Optional, Any, Dict
from tqdm.auto import tqdm
from omegaconf import DictConfig

from ...core.block.base import AbstractBaseBlock
from ...core.block.registry import register_block
from ...core.block.builder import BlockBuilder
from .state import DiffusionState
from .sampler import DiffusionSampler


@register_block("engine/loop")
class SamplingLoop(AbstractBaseBlock):
    """Sampling loop with hooks — holds sampler from config (no slots)."""

    block_type = "engine/loop"

    def __init__(self, config: DictConfig):
        super().__init__(config)
        sampler_cfg = self.config.get("sampler")
        self._sampler: Optional[DiffusionSampler] = (
            sampler_cfg if isinstance(sampler_cfg, DiffusionSampler)
            else BlockBuilder.build(sampler_cfg) if isinstance(sampler_cfg, dict) else None
        )
        self.hooks: List[Callable[[DiffusionState, int], None]] = []
        self.pre_step_hooks: List[Callable] = []
        self.post_step_hooks: List[Callable] = []
    
    def add_hook(self, hook: Callable[[DiffusionState, int], None]):
        """Добавить хук, который вызывается после каждого шага."""
        self.hooks.append(hook)
    
    def add_pre_step_hook(self, hook: Callable):
        self.pre_step_hooks.append(hook)
    
    def add_post_step_hook(self, hook: Callable):
        self.post_step_hooks.append(hook)
    
    @torch.no_grad()
    def run(
        self,
        condition: Dict[str, Any],
        num_steps: Optional[int] = None,
        initial_state: Optional[DiffusionState] = None,
        callback: Optional[Callable[[int, DiffusionState], None]] = None,
        **kwargs
    ) -> DiffusionState:
        """Запуск полного цикла."""
        sampler = self._sampler
        if sampler is None:
            raise RuntimeError("SamplingLoop: sampler not set")
        steps = num_steps or sampler.num_inference_steps
        
        # Инициализация состояния
        if initial_state is not None:
            state = initial_state
        else:
            latents = sampler._initialize_latents(kwargs.get("shape"))
            state = DiffusionState(
                latents=latents,
                timestep=torch.tensor([1.0], device=latents.device),
            )
        
        timesteps = sampler._get_timesteps(steps)
        
        for i, t in enumerate(tqdm(timesteps, desc="Sampling Loop")):
            # Pre-step hooks
            for hook in self.pre_step_hooks:
                state = hook(state, i)
            
            # Один шаг
            state.timestep = t
            state = sampler.step(state, condition, **kwargs)
            
            # Post-step hooks
            for hook in self.post_step_hooks:
                state = hook(state, i)
            
            # Пользовательский callback
            if callback is not None:
                callback(i, state)
        
        return state