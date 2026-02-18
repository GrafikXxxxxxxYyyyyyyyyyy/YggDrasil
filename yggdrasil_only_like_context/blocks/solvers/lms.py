# yggdrasil/blocks/solvers/lms.py
"""LMS (Linear Multi-Step) solver."""
import torch
from omegaconf import DictConfig
from yggdrasil.core.diffusion.solver.base import AbstractSolver
from yggdrasil.core.block.registry import register_block


@register_block("solver/lms")
class LMSSolver(AbstractSolver):
    """LMS discrete solver — multi-step linear method.
    
    Uses Adams-Bashforth linear multi-step method with sigma reparametrization.
    """
    block_type = "solver/lms"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {"type": "solver/lms"})
        self.order = int(self.config.get("order", 4))
        self._history: list = []
    
    def step(self, model_output, current_latents, timestep, process=None, **kwargs):
        next_timestep = kwargs.get("next_timestep")
        self._history.append(model_output)
        
        if len(self._history) > self.order:
            self._history.pop(0)
        
        if process is not None and hasattr(process, 'get_sigma'):
            sigma_t = process.get_sigma(timestep).view(-1, *([1] * (current_latents.dim() - 1)))
            sigma_next = process.get_sigma(next_timestep).view(-1, *([1] * (current_latents.dim() - 1))) if next_timestep is not None else torch.zeros_like(sigma_t)
            
            # Multi-step estimate
            dt = sigma_next - sigma_t
            n = len(self._history)
            if n == 1:
                noise_est = self._history[-1]
            elif n == 2:
                noise_est = (3 * self._history[-1] - self._history[-2]) / 2
            elif n == 3:
                noise_est = (23 * self._history[-1] - 16 * self._history[-2] + 5 * self._history[-3]) / 12
            else:
                noise_est = (55 * self._history[-1] - 59 * self._history[-2] + 37 * self._history[-3] - 9 * self._history[-4]) / 24
            
            return current_latents + noise_est * dt
        
        return current_latents - model_output
