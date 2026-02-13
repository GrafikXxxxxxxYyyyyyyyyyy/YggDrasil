import torch
from omegaconf import DictConfig

from .base import AbstractSolver
from ....core.block.registry import register_block


@register_block("diffusion/solver/ddim")
class DDIMSolver(AbstractSolver):
    """Классический DDIM солвер."""
    
    block_type = "diffusion/solver/ddim"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.eta = config.get("eta", 0.0)
    
    def step(self, model_output, current_latents, timestep, process, **kwargs):
        # Простая реализация DDIM (можно расширять)
        alpha = process.get_alpha(timestep) if hasattr(process, 'get_alpha') else torch.cos(timestep * 0.5 * torch.pi) ** 2
        sigma = torch.sqrt(1 - alpha)
        
        pred_x0 = process.predict_x0(model_output, current_latents, timestep)
        
        direction = torch.sqrt(1 - alpha - sigma**2) * model_output
        next_latents = torch.sqrt(alpha) * pred_x0 + direction
        
        return next_latents