from __future__ import annotations

import torch
from typing import Callable, List, Optional, Any, Dict
from tqdm.auto import tqdm
from omegaconf import DictConfig

from ...core.block.base import AbstractBaseBlock
from ...core.block.registry import register_block
from ...core.block.slot import Slot

from .state import DiffusionState
from .sampler import DiffusionSampler
from ...core.model.modular import ModularDiffusionModel


@register_block("engine/loop")
class SamplingLoop(AbstractBaseBlock):
    """Универсальный цикл сэмплирования с хуками.
    
    Используется внутри sampler и pipeline для максимальной гибкости.
    """
    
    block_type = "engine/loop"
    
    def _define_slots(self) -> Dict[str, Slot]:
        return {
            "sampler": Slot(
                name="sampler",
                accepts=DiffusionSampler,
                multiple=False,
                optional=False
            )
        }
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
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
        sampler = self._slot_children["sampler"]
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