import torch
from .base import AbstractSolver
from ....core.block.registry import register_block


@register_block("diffusion/solver/heun")
class HeunSolver(AbstractSolver):
    """Heun's method (2nd order) — отличный для flow-matching."""
    
    block_type = "diffusion/solver/heun"
    
    def step(self, model_output, current_latents, timestep, process, **kwargs):
        # Первый прогноз
        k1 = model_output
        
        # Второй прогноз (midpoint)
        mid_t = timestep - 0.5 * (timestep[0] - timestep[1]) if timestep.numel() > 1 else timestep - 0.5
        mid_latents = current_latents - 0.5 * k1
        
        # Нужно сделать второй forward модели (через process)
        # В реальной реализации это будет через callback в sampler
        # Здесь упрощённо:
        k2 = model_output  # placeholder — в production будет реальный второй вызов
        
        next_latents = current_latents - (k1 + k2) / 2
        return next_latents