# yggdrasil/blocks/solvers/pndm.py
"""PNDM (Pseudo Numerical methods for Diffusion Models) solver."""
import torch
from omegaconf import DictConfig
from yggdrasil.core.diffusion.solver.base import AbstractSolver
from yggdrasil.core.block.registry import register_block


@register_block("solver/pndm")
class PNDMSolver(AbstractSolver):
    """PNDM solver — 4th order multi-step method.
    
    Uses a linear multi-step approach. Requires history of model outputs.
    """
    block_type = "solver/pndm"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {"type": "solver/pndm"})
        self._history: list = []
        self._max_history = 4
    
    def step(self, model_output, current_latents, timestep, process=None, **kwargs):
        next_timestep = kwargs.get("next_timestep")
        self._history.append(model_output)
        
        if len(self._history) > self._max_history:
            self._history.pop(0)
        
        if process is not None and hasattr(process, 'get_alpha'):
            alpha_t = process.get_alpha(timestep)
            if next_timestep is not None:
                alpha_next = process.get_alpha(next_timestep)
            else:
                alpha_next = torch.ones_like(alpha_t)
            
            for _ in range(current_latents.dim() - 1):
                alpha_t = alpha_t.unsqueeze(-1)
                alpha_next = alpha_next.unsqueeze(-1)
            
            # Linear multi-step estimate
            if len(self._history) == 1:
                noise_est = model_output
            elif len(self._history) == 2:
                noise_est = (3 * self._history[-1] - self._history[-2]) / 2
            elif len(self._history) == 3:
                noise_est = (23 * self._history[-1] - 16 * self._history[-2] + 5 * self._history[-3]) / 12
            else:
                noise_est = (55 * self._history[-1] - 59 * self._history[-2] + 37 * self._history[-3] - 9 * self._history[-4]) / 24
            
            pred_x0 = (current_latents - (1 - alpha_t).sqrt() * noise_est) / alpha_t.sqrt().clamp(min=1e-8)
            pred_x0 = pred_x0.clamp(-20, 20)
            x_next = alpha_next.sqrt() * pred_x0 + (1 - alpha_next).sqrt() * noise_est
            return x_next
        
        return current_latents - model_output
