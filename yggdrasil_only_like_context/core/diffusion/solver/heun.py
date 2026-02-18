import torch
from typing import Optional, Callable, Any
from .base import AbstractSolver
from ....core.block.registry import register_block


@register_block("diffusion/solver/heun")
class HeunSolver(AbstractSolver):
    """Heun's method (2nd order) -- excellent for flow-matching models.
    
    Requires a `model_fn` callable in kwargs for the second evaluation.
    If not provided, falls back to first-order Euler.
    """
    
    block_type = "diffusion/solver/heun"
    
    def step(self, model_output, current_latents, timestep, process, **kwargs):
        next_timestep = kwargs.get("next_timestep")
        model_fn: Optional[Callable] = kwargs.get("model_fn")
        
        # Compute dt
        if next_timestep is not None:
            dt = next_timestep.float() - timestep.float()
        else:
            dt = torch.tensor(-1.0, device=current_latents.device, dtype=current_latents.dtype)
        
        # Reshape dt for broadcasting
        while dt.dim() < current_latents.dim():
            dt = dt.unsqueeze(-1)
        
        # First evaluation (k1)
        k1 = model_output
        
        # Euler prediction to get midpoint
        mid_latents = current_latents + k1 * dt
        
        if model_fn is not None and next_timestep is not None:
            # Real second evaluation at the endpoint (Heun's method)
            k2 = model_fn(mid_latents, next_timestep)
            # Average the two slopes (trapezoidal rule)
            next_latents = current_latents + (k1 + k2) / 2 * dt
        else:
            # Fallback to first-order Euler when no model_fn available
            next_latents = mid_latents
        
        return next_latents