# yggdrasil/core/diffusion/ddpm.py
"""DDPM-процесс для SD 1.5 / SDXL — линейное бета-расписание, 1000 дискретных шагов."""
from __future__ import annotations

import torch
from omegaconf import DictConfig

from ...core.block.base import AbstractBlock
from ...core.block.registry import register_block
from .process import AbstractDiffusionProcess


@register_block("diffusion/process/ddpm")
class DDPMProcess(AbstractDiffusionProcess):
    """DDPM с линейным бета-расписанием (как в SD 1.5).

    Таймстепы: целые числа 0..999, где 999 = максимальный шум, 0 = почти чистый.
    alpha_cumprod вычисляется один раз при инициализации.
    """

    block_type = "diffusion/process/ddpm"

    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {})
        # Линейное бета-расписание как в оригинальном SD 1.5
        num_train_timesteps = int(self.config.get("num_train_timesteps", 1000))
        beta_start = float(self.config.get("beta_start", 0.00085))
        beta_end = float(self.config.get("beta_end", 0.012))

        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps) ** 2
        alphas = 1.0 - betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.num_train_timesteps = num_train_timesteps

    def get_alpha_cumprod(self, t: torch.Tensor) -> torch.Tensor:
        """alpha_cumprod для дискретного таймстепа t (int, 0..999)."""
        t = t.long().clamp(0, self.num_train_timesteps - 1)
        return self.alphas_cumprod[t]

    # Совместимость с солвером
    get_alpha = get_alpha_cumprod

    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.get_alpha(t)).sqrt()

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        alpha = self.get_alpha(t).view(-1, *([1] * (x0.ndim - 1)))
        return {
            "xt": alpha.sqrt() * x0 + (1 - alpha).sqrt() * noise,
            "noise": noise,
        }

    def predict_x0(
        self, model_output: torch.Tensor, xt: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """pred_x0 из предсказанного шума (epsilon-параметризация)."""
        alpha = self.get_alpha(t).to(dtype=xt.dtype)
        alpha = alpha.view(-1, *([1] * (xt.ndim - 1)))
        return (xt - (1 - alpha).sqrt() * model_output) / alpha.sqrt().clamp(min=1e-8)

    def predict_velocity(self, model_output, xt, t):
        return model_output

    def reverse_step(self, model_output, xt, t, **kwargs):
        pred_x0 = self.predict_x0(model_output, xt, t)
        alpha = self.get_alpha(t).view(-1, *([1] * (xt.ndim - 1)))
        return alpha.sqrt() * pred_x0 + (1 - alpha).sqrt() * model_output
