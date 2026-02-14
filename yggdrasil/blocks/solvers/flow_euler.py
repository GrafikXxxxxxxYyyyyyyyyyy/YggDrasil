# yggdrasil/blocks/solvers/flow_euler.py
"""Flow Matching Euler solver (for Flux, SD3, Rectified Flow models)."""
import torch
from omegaconf import DictConfig
from yggdrasil.core.diffusion.solver.base import AbstractSolver
from yggdrasil.core.block.registry import register_block


@register_block("solver/flow_euler")
class FlowEulerSolver(AbstractSolver):
    """Flow Matching Euler solver.
    
    Simple but effective solver for flow-matching models (Flux, SD3).
    Uses continuous timesteps [0, 1] and velocity predictions.
    """
    block_type = "solver/flow_euler"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {"type": "solver/flow_euler"})
    
    def step(self, model_output, current_latents, timestep, process=None, **kwargs):
        next_timestep = kwargs.get("next_timestep")
        
        if next_timestep is not None:
            t = timestep.float()
            t_next = next_timestep.float()
            while t.dim() < current_latents.dim():
                t = t.unsqueeze(-1)
                t_next = t_next.unsqueeze(-1)
            dt = t_next - t
        else:
            dt = torch.tensor(-0.02, device=current_latents.device, dtype=current_latents.dtype)
            while dt.dim() < current_latents.dim():
                dt = dt.unsqueeze(-1)
        
        # Euler step: x_{t+1} = x_t + v(x_t, t) * dt
        return current_latents + model_output * dt
