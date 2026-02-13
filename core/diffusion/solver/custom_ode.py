import torch
from .base import AbstractSolver
from ....core.block.registry import register_block


@register_block("diffusion/solver/custom_ode")
class CustomODESolver(AbstractSolver):
    """Шаблон для любого ODE/SDE солвера (Dopri5, Tsit5, SDE и т.д.)."""
    
    block_type = "diffusion/solver/custom_ode"
    
    def step(self, model_output, current_latents, timestep, process, **kwargs):
        # Пользователь может переопределить через hook
        # Или реализовать через torchdiffeq / diffrax
        return current_latents - model_output * (timestep[0] - timestep[1])