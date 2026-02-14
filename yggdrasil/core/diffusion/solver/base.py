import torch
from typing import Any, Dict

from ....core.block.base import AbstractBlock
from ....core.block.registry import register_block
from ....core.block.port import Port, InputPort, OutputPort, TensorSpec
from ..process import AbstractDiffusionProcess


@register_block("diffusion/solver/abstract")
class AbstractSolver(AbstractBlock):
    """Абстрактный солвер (DDIM, Heun, Euler, DPM и т.д.).
    
    Контракт: реализовать ``process()`` или ``step()``.
    
    Порты:
        IN:  model_output, current_latents, timestep, next_timestep (opt), process (opt)
        OUT: next_latents
    
    Пример::
    
        @register_block("solver/my_solver")
        class MySolver(AbstractSolver):
            block_type = "solver/my_solver"
            def step(self, model_output, current_latents, timestep, process, **kw):
                return current_latents - 0.1 * model_output
    """
    
    block_type = "diffusion/solver/abstract"
    
    @classmethod
    def declare_io(cls) -> dict:
        return {
            "model_output": InputPort("model_output", spec=TensorSpec(space="latent"), description="Model prediction"),
            "current_latents": InputPort("current_latents", spec=TensorSpec(space="latent"), description="Current noisy latents"),
            "timestep": InputPort("timestep", data_type="tensor", description="Current timestep"),
            "next_timestep": InputPort("next_timestep", data_type="tensor", optional=True, description="Next timestep"),
            "process": InputPort("process", data_type="any", optional=True, description="DiffusionProcess for alpha/sigma schedule"),
            "next_latents": OutputPort("next_latents", spec=TensorSpec(space="latent"), description="Denoised latents"),
        }
    
    def process(self, **port_inputs) -> dict:
        model_output = port_inputs.get("model_output")
        current_latents = port_inputs.get("current_latents")
        timestep = port_inputs.get("timestep")
        process = port_inputs.get("process")
        
        if current_latents is None:
            return {"next_latents": model_output, "output": model_output, "latents": model_output}
        
        extra = {k: v for k, v in port_inputs.items()
                 if k not in ("model_output", "current_latents", "timestep", "process")}
        result = self.step(model_output, current_latents, timestep, process, **extra)
        return {"next_latents": result, "output": result, "latents": result}
    
    def _define_slots(self):
        return {}

    def _forward_impl(self, *args, **kwargs):
        return None
    
    def step(
        self,
        model_output: torch.Tensor,
        current_latents: torch.Tensor,
        timestep: torch.Tensor,
        process: AbstractDiffusionProcess,
        **kwargs: Any
    ) -> torch.Tensor:
        """One solver step. Override this or process()."""
        raise NotImplementedError(
            f"{type(self).__name__} должен реализовать process() или step()"
        )