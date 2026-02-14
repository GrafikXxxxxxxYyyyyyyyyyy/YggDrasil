# yggdrasil/blocks/solvers/deis.py
"""DEIS (Diffusion Exponential Integrator Sampler)."""
import torch
from omegaconf import DictConfig
from yggdrasil.core.diffusion.solver.base import AbstractSolver
from yggdrasil.core.block.registry import register_block


@register_block("solver/deis")
class DEISSolver(AbstractSolver):
    """DEIS solver — exponential integrator for diffusion ODEs.
    
    Uses polynomial approximation of the data prediction model.
    """
    block_type = "solver/deis"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {"type": "solver/deis"})
        self.order = int(self.config.get("order", 3))
        self._history: list = []
    
    def step(self, model_output, current_latents, timestep, process=None, **kwargs):
        next_timestep = kwargs.get("next_timestep")
        self._history.append(model_output)
        
        if len(self._history) > self.order:
            self._history.pop(0)
        
        if process is not None and hasattr(process, 'get_alpha'):
            alpha_t = process.get_alpha(timestep)
            sigma_t = process.get_sigma(timestep)
            
            if next_timestep is not None:
                alpha_next = process.get_alpha(next_timestep)
                sigma_next = process.get_sigma(next_timestep)
            else:
                alpha_next = torch.ones_like(alpha_t)
                sigma_next = torch.zeros_like(sigma_t)
            
            for _ in range(current_latents.dim() - 1):
                alpha_t = alpha_t.unsqueeze(-1)
                sigma_t = sigma_t.unsqueeze(-1)
                alpha_next = alpha_next.unsqueeze(-1)
                sigma_next = sigma_next.unsqueeze(-1)
            
            lambda_t = torch.log(alpha_t / sigma_t)
            lambda_next = torch.log(alpha_next / sigma_next)
            h = lambda_next - lambda_t
            
            # Exponential integrator step
            x_next = (alpha_next / alpha_t) * current_latents - sigma_next * (torch.exp(h) - 1) * model_output
            
            return x_next
        
        return current_latents - model_output
