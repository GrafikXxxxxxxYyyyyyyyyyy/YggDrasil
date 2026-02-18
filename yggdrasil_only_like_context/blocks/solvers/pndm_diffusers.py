# yggdrasil/blocks/solvers/pndm_diffusers.py
"""PNDMSolver — обёртка над diffusers PNDMScheduler для консистентности с StableDiffusionPipeline.

Дефолтный планировщик runwayml/stable-diffusion-v1-5 — PNDM. Использование этого блока
даёт тот же результат, что и pipe(prompt, num_inference_steps=28, generator=...).
"""
from __future__ import annotations

import torch
from omegaconf import DictConfig
from typing import Any, Dict

from yggdrasil.core.diffusion.solver.base import AbstractSolver
from yggdrasil.core.block.registry import register_block


@register_block("solver/pndm_diffusers")
class PNDMDiffusersSolver(AbstractSolver):
    """PNDM solver — обёртка над diffusers PNDMScheduler (паритет с StableDiffusionPipeline)."""

    block_type = "solver/pndm_diffusers"

    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "solver/pndm_diffusers"}
        super().__init__(config)
        from diffusers import PNDMScheduler

        self.scheduler = PNDMScheduler(
            num_train_timesteps=int(self.config.get("num_train_timesteps", 1000)),
            beta_start=float(self.config.get("beta_start", 0.00085)),
            beta_end=float(self.config.get("beta_end", 0.012)),
            beta_schedule=self.config.get("beta_schedule", "scaled_linear"),
            skip_prk_steps=self.config.get("skip_prk_steps", True),
            set_alpha_to_one=self.config.get("set_alpha_to_one", False),
            steps_offset=int(self.config.get("steps_offset", 1)),
        )
        self._num_inference_steps = None
        self._device = None

    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        self._num_inference_steps = num_inference_steps
        self._device = device
        self.scheduler.set_timesteps(num_inference_steps, device=device)

    @classmethod
    def declare_io(cls) -> dict:
        from yggdrasil.core.block.port import InputPort, OutputPort, TensorSpec

        return {
            "model_output": InputPort("model_output", spec=TensorSpec(space="latent")),
            "current_latents": InputPort("current_latents", spec=TensorSpec(space="latent")),
            "timestep": InputPort("timestep", data_type="tensor"),
            "next_timestep": InputPort("next_timestep", data_type="tensor", optional=True),
            "num_steps": InputPort("num_steps", data_type="any", optional=True),
            "next_latents": OutputPort("next_latents", spec=TensorSpec(space="latent")),
        }

    def process(self, **port_inputs) -> Dict[str, Any]:
        model_output = port_inputs.get("model_output")
        current_latents = port_inputs.get("current_latents")
        timestep = port_inputs.get("timestep")
        num_steps = port_inputs.get("num_steps")

        if current_latents is None:
            return {"next_latents": model_output, "output": model_output, "latents": model_output}

        device = current_latents.device
        orig_dtype = current_latents.dtype
        if num_steps is not None and (self._num_inference_steps != num_steps or self._device != device):
            self.set_timesteps(int(num_steps), device)
        elif self._num_inference_steps is None or self._device != device:
            self.set_timesteps(getattr(self, "_cached_num_steps", 28), device)

        t = timestep.float()
        if t.dim() > 0 and t.numel() > 1:
            t = t[0]
        t_scalar = t.item() if t.numel() == 1 else float(t)
        # PNDMScheduler.step() expects integer timestep for internal indexing
        t_int = int(round(t_scalar))

        current_latents = current_latents.to(device=device, dtype=torch.float32)
        model_output = model_output.to(device=device, dtype=torch.float32)
        out = self.scheduler.step(
            model_output=model_output,
            timestep=t_int,
            sample=current_latents,
        )
        prev_sample = out.prev_sample.to(orig_dtype)
        return {"next_latents": prev_sample, "output": prev_sample, "latents": prev_sample}
