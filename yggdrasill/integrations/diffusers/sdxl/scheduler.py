"""SDXL scheduler nodes (reuses SD1.5 logic with SDXL defaults)."""
from __future__ import annotations

from typing import Any, Dict

from yggdrasill.integrations.diffusers.sd15.scheduler import (
    SD15SchedulerSetupNode,
    SD15SchedulerStepNode,
)


class SDXLSchedulerSetupNode(SD15SchedulerSetupNode):
    """SDXL scheduler setup with support for denoising_start/end slicing."""

    @property
    def block_type(self) -> str:
        return "sdxl/scheduler_setup"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = super().forward(inputs)

        denoising_start = self._config.get("denoising_start")
        denoising_end = self._config.get("denoising_end")

        if denoising_start is not None or denoising_end is not None:
            timesteps = result["timesteps"]
            total = len(timesteps)
            start_idx = int(total * (denoising_start or 0.0))
            end_idx = int(total * (denoising_end or 1.0))
            result["timesteps"] = timesteps[start_idx:end_idx]

        return result


class SDXLSchedulerStepNode(SD15SchedulerStepNode):
    """SDXL scheduler step (identical interface to SD1.5)."""

    @property
    def block_type(self) -> str:
        return "sdxl/scheduler_step"
