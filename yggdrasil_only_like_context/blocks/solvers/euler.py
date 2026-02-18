# yggdrasil/blocks/solvers/euler.py
"""Euler and Euler Ancestral solvers."""
import torch
from omegaconf import DictConfig
from yggdrasil.core.diffusion.solver.base import AbstractSolver
from yggdrasil.core.block.registry import register_block


@register_block("solver/euler")
class EulerSolver(AbstractSolver):
    """Euler discrete solver (deterministic, 1st order).
    
    sigma(t) reparametrization. Used by many diffusers models.
    """
    block_type = "solver/euler"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {"type": "solver/euler"})
    
    def step(self, model_output, current_latents, timestep, process=None, **kwargs):
        next_timestep = kwargs.get("next_timestep")
        
        if process is not None and hasattr(process, 'get_sigma'):
            sigma_t = process.get_sigma(timestep).view(-1, *([1] * (current_latents.dim() - 1)))
            sigma_next = process.get_sigma(next_timestep).view(-1, *([1] * (current_latents.dim() - 1))) if next_timestep is not None else torch.zeros_like(sigma_t)
            dt = sigma_next - sigma_t
            return current_latents + model_output * dt
        
        # Fallback: simple Euler step
        dt = -1.0 if next_timestep is None else float(next_timestep - timestep)
        return current_latents + model_output * dt


@register_block("solver/euler_ancestral")
class EulerAncestralSolver(AbstractSolver):
    """Euler Ancestral solver (stochastic, adds noise at each step)."""
    block_type = "solver/euler_ancestral"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {"type": "solver/euler_ancestral"})
        self.eta = float(self.config.get("eta", 1.0))
    
    def step(self, model_output, current_latents, timestep, process=None, **kwargs):
        next_timestep = kwargs.get("next_timestep")
        
        if process is not None and hasattr(process, 'get_sigma'):
            sigma_t = process.get_sigma(timestep).view(-1, *([1] * (current_latents.dim() - 1)))
            sigma_next = process.get_sigma(next_timestep).view(-1, *([1] * (current_latents.dim() - 1))) if next_timestep is not None else torch.zeros_like(sigma_t)
            
            sigma_up = (sigma_next ** 2 * (sigma_t ** 2 - sigma_next ** 2) / sigma_t ** 2).clamp(min=0).sqrt() * self.eta
            sigma_down = (sigma_next ** 2 - sigma_up ** 2).clamp(min=0).sqrt()
            
            dt = sigma_down - sigma_t
            x_next = current_latents + model_output * dt
            
            if sigma_up.sum() > 0:
                noise = torch.randn_like(current_latents)
                x_next = x_next + sigma_up * noise
            
            return x_next
        
        return current_latents + model_output * (-1.0)
