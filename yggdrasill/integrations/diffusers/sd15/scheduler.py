"""SD1.5 scheduler nodes: setup + per-step transition."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.diffusion import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractOuterModule, AbstractInnerModule


class SD15SchedulerSetupNode(AbstractOuterModule):
    """Configures scheduler timesteps and provides initial scheduler state.

    This is an outer module that runs once before the denoising loop.
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        scheduler: Any = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)
        self._scheduler = scheduler

    @property
    def block_type(self) -> str:
        return "sd15/scheduler_setup"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.ANY, optional=True),
            Port(C.PORT_TIMESTEPS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_SCHEDULER_STATE, PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        num_steps = self._config.get("num_inference_steps", 50)
        custom_timesteps = self._config.get("timesteps")
        custom_sigmas = self._config.get("sigmas")
        device = self._config.get("device", "cpu")

        kwargs: Dict[str, Any] = {}
        if custom_timesteps is not None:
            kwargs["timesteps"] = custom_timesteps
        elif custom_sigmas is not None:
            kwargs["sigmas"] = custom_sigmas

        self._scheduler.set_timesteps(num_steps, device=device, **kwargs)

        return {
            C.PORT_TIMESTEPS: self._scheduler.timesteps,
            C.PORT_SCHEDULER_STATE: {
                "scheduler": self._scheduler,
                "init_noise_sigma": getattr(self._scheduler, "init_noise_sigma", 1.0),
                "order": getattr(self._scheduler, "order", 1),
            },
        }


class SD15SchedulerStepNode(AbstractInnerModule):
    """Performs one scheduler step: scale_model_input + step.

    This is an inner module inside the denoising loop.
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        scheduler: Any = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)
        self._scheduler = scheduler

    @property
    def block_type(self) -> str:
        return "sd15/scheduler_step"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_LATENTS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_TIMESTEP, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_NOISE_PRED, PortDirection.IN, PortType.TENSOR),
            Port("next_latent", PortDirection.OUT, PortType.TENSOR),
            Port("next_timestep", PortDirection.OUT, PortType.TENSOR, optional=True),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        latents = inputs[C.PORT_LATENTS]
        timestep = inputs[C.PORT_TIMESTEP]
        noise_pred = inputs[C.PORT_NOISE_PRED]

        eta = self._config.get("eta", 0.0)
        generator = self._config.get("generator")

        step_kwargs: Dict[str, Any] = {}
        if eta > 0:
            step_kwargs["eta"] = eta
        if generator is not None:
            step_kwargs["generator"] = generator

        result = self._scheduler.step(noise_pred, timestep, latents, **step_kwargs)
        next_latents = result.prev_sample

        return {
            "next_latent": next_latents,
            "next_timestep": timestep,
        }

    def scale_model_input(self, latents: Any, timestep: Any) -> Any:
        """Public helper for nodes that need to scale before UNet."""
        return self._scheduler.scale_model_input(latents, timestep)
