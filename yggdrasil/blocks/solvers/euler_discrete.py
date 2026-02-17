# yggdrasil/blocks/solvers/euler_discrete.py
"""EulerDiscreteSolver — parity with diffusers EulerDiscreteScheduler.

SDXL and many diffusers pipelines use EulerDiscreteScheduler by default, NOT DDIM.
Key differences from DDIM:
1. scale_model_input: UNet input must be scaled by 1/(sigma^2+1)^0.5
2. init_noise_sigma: for "leading" spacing = (max_sigma^2+1)^0.5
3. Step formula uses sigma parametrization (Euler method)

This block wraps diffusers.EulerDiscreteScheduler for exact parity.
"""
from __future__ import annotations

import torch
from omegaconf import DictConfig
from typing import Any, Dict

from yggdrasil.core.diffusion.solver.base import AbstractSolver
from yggdrasil.core.block.registry import register_block


@register_block("solver/euler_discrete")
class EulerDiscreteSolver(AbstractSolver):
    """Euler Discrete solver matching diffusers EulerDiscreteScheduler.

    Requires: scale_model_input() before UNet, and init_noise_sigma for initial noise.
    Timesteps must be float (from scheduler.timesteps), not integer indices.
    """

    block_type = "solver/euler_discrete"

    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "solver/euler_discrete"}
        super().__init__(config)
        from diffusers import EulerDiscreteScheduler

        pretrained = self.config.get("pretrained")
        if pretrained:
            self.scheduler = EulerDiscreteScheduler.from_pretrained(
                pretrained, subfolder="scheduler"
            )
        else:
            num_train_timesteps = int(self.config.get("num_train_timesteps", 1000))
            beta_start = float(self.config.get("beta_start", 0.00085))
            beta_end = float(self.config.get("beta_end", 0.012))
            beta_schedule = self.config.get("beta_schedule", "scaled_linear")
            steps_offset = int(self.config.get("steps_offset", 1))
            timestep_spacing = self.config.get("timestep_spacing", "leading")
            self.scheduler = EulerDiscreteScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                steps_offset=steps_offset,
                timestep_spacing=timestep_spacing,
            )
        self._num_inference_steps = None
        self._device = None

    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        """Call before loop. Sets timesteps and sigmas for the run."""
        self._num_inference_steps = num_inference_steps
        self._device = device
        self.scheduler.set_timesteps(num_inference_steps, device=device)

    def scale_model_input(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Scale latents before UNet. Must be called each step before backbone."""
        t = timestep.float()
        if t.dim() == 0:
            t = t.unsqueeze(0)
        return self.scheduler.scale_model_input(sample, t)

    @property
    def init_noise_sigma(self) -> float:
        """Scale for initial noise. InferencePipeline should multiply noise by this."""
        s = self.scheduler.init_noise_sigma
        return float(s) if hasattr(s, "item") else float(s)

    @classmethod
    def declare_io(cls) -> dict:
        from yggdrasil.core.block.port import InputPort, OutputPort, TensorSpec

        return {
            "model_output": InputPort("model_output", spec=TensorSpec(space="latent")),
            "current_latents": InputPort("current_latents", spec=TensorSpec(space="latent")),
            "timestep": InputPort("timestep", data_type="tensor"),
            "next_timestep": InputPort("next_timestep", data_type="tensor", optional=True),
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
            n = getattr(self, "_cached_num_steps", 28)
            self.set_timesteps(n, device)

        t = timestep.float()
        if t.dim() > 0 and t.numel() > 1:
            t = t[0]
        t_scalar = t.item() if t.numel() == 1 else float(t)

        # Step in float32 for numerical stability (same as DDIMSolver); return in original dtype
        current_latents = current_latents.to(device=device, dtype=torch.float32)
        model_output = model_output.to(device=device, dtype=torch.float32)
        out = self.scheduler.step(
            model_output=model_output,
            timestep=t_scalar,
            sample=current_latents,
        )
        prev_sample = out.prev_sample.to(orig_dtype)
        return {"next_latents": prev_sample, "output": prev_sample, "latents": prev_sample}

    def step(self, model_output, current_latents, timestep, process=None, **kwargs) -> torch.Tensor:
        return self.process(
            model_output=model_output,
            current_latents=current_latents,
            timestep=timestep,
            **kwargs,
        )["next_latents"]
