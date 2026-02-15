# yggdrasil/blocks/solvers/flow_euler.py
"""Flow Matching Euler solver (for Flux, SD3, Rectified Flow models)."""
import torch
from omegaconf import DictConfig
from yggdrasil.core.diffusion.solver.base import AbstractSolver
from yggdrasil.core.block.registry import register_block


@register_block("solver/flow_euler")
class FlowEulerSolver(AbstractSolver):
    """Flow Matching Euler solver.

    For SD3: delegates to Diffusers FlowMatchEulerDiscreteScheduler.step() when
    num_steps is provided (first step), so dt and casting match Diffusers exactly.
    Fallback: manual Euler with dt = (t_next - t) / num_train_timesteps and float32.
    """
    block_type = "solver/flow_euler"

    @classmethod
    def declare_io(cls):
        from ...core.block.port import InputPort
        base = dict(super().declare_io())
        base["num_steps"] = InputPort("num_steps", data_type="int", optional=True, description="Number of inference steps (used to init scheduler for SD3)")
        return base

    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {"type": "solver/flow_euler"})
        self._scheduler = None

    def step(self, model_output, current_latents, timestep, process=None, **kwargs):
        num_steps = kwargs.get("num_steps")
        if num_steps is not None and self._scheduler is None:
            try:
                from diffusers import FlowMatchEulerDiscreteScheduler
                pretrained = self.config.get("scheduler_pretrained")
                if pretrained:
                    # Load scheduler from model repo (SD3 uses shift=3.0; default would be wrong)
                    self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                        pretrained, subfolder="scheduler"
                    )
                else:
                    self._scheduler = FlowMatchEulerDiscreteScheduler()
                self._scheduler.set_timesteps(num_steps, device=current_latents.device)
            except Exception:
                pass

        if self._scheduler is not None:
            # Use exact Diffusers step (same dt, float32 upcast).
            # FlowMatchEulerDiscreteScheduler.step() requires float timestep (rejects int/long).
            timestep_val = timestep.float().to(current_latents.device)
            if timestep_val.dim() == 0:
                timestep_val = timestep_val.unsqueeze(0)
            out = self._scheduler.step(
                model_output, timestep_val, current_latents, return_dict=True
            )
            return out.prev_sample

        # Fallback: manual Euler (e.g. Flux or when diffusers not available)
        next_timestep = kwargs.get("next_timestep")
        num_train_timesteps = int(getattr(self, "num_train_timesteps", 1000))

        if next_timestep is not None:
            t = timestep.float().to(current_latents.device)
            t_next = next_timestep.float().to(current_latents.device)
            if t.numel() and t.max().item() > 2:
                dt = (t_next - t) / num_train_timesteps
            else:
                dt = t_next - t
            while dt.dim() < current_latents.dim():
                dt = dt.unsqueeze(-1)
        else:
            dt = torch.tensor(-0.02, device=current_latents.device, dtype=torch.float32)
            while dt.dim() < current_latents.dim():
                dt = dt.unsqueeze(-1)

        orig_dtype = current_latents.dtype
        sample = current_latents.to(torch.float32)
        model_output = model_output.to(torch.float32)
        next_latents = sample + model_output * dt
        return next_latents.to(orig_dtype)
