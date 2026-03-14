"""SD1.5 latent initialization nodes for text2img, img2img, inpaint."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.diffusion import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractOuterModule


class SD15LatentInitNode(AbstractOuterModule):
    """Initializes latents for text2img (random noise) or img2img (encoded + noise).

    For text2img: generates random noise scaled by scheduler.init_noise_sigma.
    For img2img: receives pre-encoded latents on ``init_latents``, adds noise.
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)

    @property
    def block_type(self) -> str:
        return "sd15/latent_init"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_SCHEDULER_STATE, PortDirection.IN, PortType.ANY, optional=True),
            Port(C.PORT_INIT_LATENTS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_LATENTS, PortDirection.OUT, PortType.TENSOR),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        existing_latents = inputs.get(C.PORT_INIT_LATENTS)
        sched_state = inputs.get(C.PORT_SCHEDULER_STATE, {})
        init_noise_sigma = sched_state.get("init_noise_sigma", 1.0) if isinstance(sched_state, dict) else 1.0

        if existing_latents is not None:
            latents = existing_latents * init_noise_sigma
            return {C.PORT_LATENTS: latents}

        height = self._config.get("height", 512)
        width = self._config.get("width", 512)
        batch_size = self._config.get("batch_size", 1)
        num_channels = self._config.get("num_latent_channels", 4)
        device = self._config.get("device", "cpu")
        dtype_str = self._config.get("dtype", "float32")

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        dtype = dtype_map.get(dtype_str, torch.float32)

        shape = (batch_size, num_channels, height // 8, width // 8)

        generator = None
        seed = self._config.get("seed")
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        latents = torch.randn(shape, generator=generator, device="cpu", dtype=dtype)
        latents = latents.to(device) * init_noise_sigma

        return {C.PORT_LATENTS: latents}
