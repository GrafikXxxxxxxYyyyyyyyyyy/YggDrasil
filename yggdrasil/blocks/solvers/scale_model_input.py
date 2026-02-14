# yggdrasil/blocks/solvers/scale_model_input.py
"""ScaleModelInput — delegates to EulerDiscreteSolver.scale_model_input.

Must use the SAME scheduler instance as the step() call, so that:
1. sigma values match exactly (interpolated inference schedule)
2. is_scale_input_called is set before step() to avoid diffusers warning
"""
from __future__ import annotations

from typing import Any, Dict

from omegaconf import DictConfig

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.base import AbstractBlock
from yggdrasil.core.block.port import InputPort, OutputPort, TensorSpec


@register_block("solver/scale_model_input")
class ScaleModelInputBlock(AbstractBlock):
    """Calls solver.scale_model_input() — must receive the same solver instance as the step.

    Config: "solver" — the EulerDiscreteSolver block (passed at build time).
    """

    block_type = "solver/scale_model_input"

    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {})
        self._solver = config.get("solver") if config else None

    def set_solver(self, solver):
        """Inject solver reference (call after graph build when solver is available)."""
        self._solver = solver

    @classmethod
    def declare_io(cls) -> dict:
        return {
            "sample": InputPort("sample", spec=TensorSpec(space="latent")),
            "timestep": InputPort("timestep", data_type="tensor"),
            "num_steps": InputPort("num_steps", data_type="any", optional=True),
            "scaled": OutputPort("scaled", spec=TensorSpec(space="latent")),
        }

    def process(self, **port_inputs) -> Dict[str, Any]:
        sample = port_inputs["sample"]
        timestep = port_inputs["timestep"]
        num_steps = port_inputs.get("num_steps")
        solver = self._solver
        if solver is None or not hasattr(solver, "scale_model_input"):
            return {"scaled": sample, "output": sample}
        if num_steps is not None and hasattr(solver, "set_timesteps"):
            solver.set_timesteps(int(num_steps), sample.device)
        scaled = solver.scale_model_input(sample, timestep)
        return {"scaled": scaled, "output": scaled}
