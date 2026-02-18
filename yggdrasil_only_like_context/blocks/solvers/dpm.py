# yggdrasil/blocks/solvers/dpm.py
"""DPM++ solvers (DPM++2M, DPM++ SDE)."""
import torch
from omegaconf import DictConfig
from yggdrasil.core.diffusion.solver.base import AbstractSolver
from yggdrasil.core.block.registry import register_block


@register_block("solver/dpm_pp_2m")
class DPMPlusPlusTwoM(AbstractSolver):
    """DPM++2M multistep solver (2nd order, deterministic).
    
    One of the best solvers for diffusion models. Fast convergence.
    """
    block_type = "solver/dpm_pp_2m"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {"type": "solver/dpm_pp_2m"})
        self._prev_model_output = None
    
    def step(self, model_output, current_latents, timestep, process=None, **kwargs):
        next_timestep = kwargs.get("next_timestep")
        
        if process is not None and hasattr(process, 'get_alpha'):
            alpha_t = process.get_alpha(timestep)
            sigma_t = process.get_sigma(timestep)
            
            if next_timestep is not None:
                alpha_next = process.get_alpha(next_timestep)
                sigma_next = process.get_sigma(next_timestep)
            else:
                alpha_next = torch.ones_like(alpha_t)
                sigma_next = torch.zeros_like(sigma_t)
            
            # Reshape for broadcasting
            for _ in range(current_latents.dim() - 1):
                alpha_t = alpha_t.unsqueeze(-1)
                sigma_t = sigma_t.unsqueeze(-1)
                alpha_next = alpha_next.unsqueeze(-1)
                sigma_next = sigma_next.unsqueeze(-1)
            
            lambda_t = torch.log(alpha_t / sigma_t)
            lambda_next = torch.log(alpha_next / sigma_next)
            h = lambda_next - lambda_t
            
            if self._prev_model_output is not None:
                # 2nd order step
                r = h / (lambda_t - torch.log(alpha_t / sigma_t))
                D0 = model_output
                D1 = (model_output - self._prev_model_output) / (2 * r)
                x_next = (sigma_next / sigma_t) * current_latents - alpha_next * (torch.exp(-h) - 1) * (D0 + D1)
            else:
                # 1st order step (first iteration)
                x_next = (sigma_next / sigma_t) * current_latents - alpha_next * (torch.exp(-h) - 1) * model_output
            
            self._prev_model_output = model_output
            return x_next
        
        # Fallback
        self._prev_model_output = model_output
        return current_latents - model_output


@register_block("solver/dpm_pp_sde")
class DPMPlusPlusSDE(AbstractSolver):
    """DPM++ SDE solver (stochastic variant with better diversity)."""
    block_type = "solver/dpm_pp_sde"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {"type": "solver/dpm_pp_sde"})
        self.noise_scale = float(self.config.get("noise_scale", 1.0))
    
    def step(self, model_output, current_latents, timestep, process=None, **kwargs):
        next_timestep = kwargs.get("next_timestep")
        
        if process is not None and hasattr(process, 'get_sigma'):
            sigma_t = process.get_sigma(timestep).view(-1, *([1] * (current_latents.dim() - 1)))
            sigma_next = process.get_sigma(next_timestep).view(-1, *([1] * (current_latents.dim() - 1))) if next_timestep is not None else torch.zeros_like(sigma_t)
            
            # Deterministic part
            x_next = current_latents + model_output * (sigma_next - sigma_t)
            
            # Stochastic part
            if sigma_next.sum() > 0:
                noise = torch.randn_like(current_latents) * self.noise_scale
                x_next = x_next + (sigma_next * (1 - (sigma_t / sigma_next) ** 2).clamp(min=0).sqrt()) * noise
            
            return x_next
        
        return current_latents - model_output
