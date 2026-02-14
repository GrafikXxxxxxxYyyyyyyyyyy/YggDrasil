import math
import torch
from omegaconf import DictConfig

from .base import AbstractSolver
from ....core.block.registry import register_block


@register_block("diffusion/solver/ddim")
class DDIMSolver(AbstractSolver):
    """DDIM солвер для дискретных таймстепов.
    
    Работает в двух режимах:
    1. С DiffusionProcess: использует точные alpha из process
    2. Без DiffusionProcess (graph mode fallback): вычисляет alpha из cosine schedule
    
    Формула (eta=0, детерминистический):
        x_{t-1} = sqrt(alpha_{t-1}) * pred_x0
                + sqrt(1 - alpha_{t-1}) * predicted_noise
    """
    
    block_type = "diffusion/solver/ddim"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.eta = config.get("eta", 0.0)
        self.num_train_timesteps = int(config.get("num_train_timesteps", 1000))
    
    def _cosine_alpha(self, timestep: torch.Tensor) -> torch.Tensor:
        """Fallback cosine alpha schedule when no DiffusionProcess available."""
        # Normalize timestep to [0, 1]
        t = timestep.float()
        if t.max() > 1.0:
            t = t / self.num_train_timesteps
        # Cosine schedule: alpha_bar(t) = cos^2(pi/2 * t)
        return torch.cos(t * math.pi / 2.0) ** 2
    
    def step(self, model_output, current_latents, timestep, process=None, **kwargs):
        next_timestep = kwargs.get("next_timestep")

        # Get alpha cumprod — from process or fallback
        if process is not None and hasattr(process, 'get_alpha'):
            alpha_prod_t = process.get_alpha(timestep).to(dtype=current_latents.dtype)
            if next_timestep is not None:
                alpha_prod_t_prev = process.get_alpha(next_timestep).to(dtype=current_latents.dtype)
            else:
                alpha_prod_t_prev = torch.ones_like(alpha_prod_t)
        else:
            # Fallback: cosine schedule
            alpha_prod_t = self._cosine_alpha(timestep).to(
                device=current_latents.device, dtype=current_latents.dtype
            )
            if next_timestep is not None:
                alpha_prod_t_prev = self._cosine_alpha(next_timestep).to(
                    device=current_latents.device, dtype=current_latents.dtype
                )
            else:
                alpha_prod_t_prev = torch.ones_like(alpha_prod_t)

        # Reshape для broadcasting
        while alpha_prod_t.dim() < current_latents.dim():
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
            alpha_prod_t_prev = alpha_prod_t_prev.unsqueeze(-1)

        # Предсказание x0 из epsilon
        pred_x0 = (current_latents - (1 - alpha_prod_t).sqrt() * model_output) / alpha_prod_t.sqrt().clamp(min=1e-8)
        pred_x0 = pred_x0.clamp(-20, 20)

        # DDIM шаг (eta=0 → детерминистический)
        pred_direction = (1 - alpha_prod_t_prev).sqrt() * model_output
        next_latents = alpha_prod_t_prev.sqrt() * pred_x0 + pred_direction

        # Стохастическая часть (eta > 0)
        if self.eta > 0:
            variance = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) * (1 - alpha_prod_t / alpha_prod_t_prev)
            sigma = self.eta * variance.clamp(min=0).sqrt()
            noise = torch.randn_like(current_latents)
            next_latents = next_latents + sigma * noise

        return next_latents
