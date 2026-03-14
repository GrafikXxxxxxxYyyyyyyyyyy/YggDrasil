"""SDXL latent initialization (1024x1024 defaults)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from yggdrasill.integrations.diffusers.sd15.latent_init import SD15LatentInitNode


class SDXLLatentInitNode(SD15LatentInitNode):
    """Initializes latents for SDXL (1024x1024 default)."""

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = dict(config or {})
        cfg.setdefault("height", 1024)
        cfg.setdefault("width", 1024)
        super().__init__(node_id=node_id, block_id=block_id, config=cfg)

    @property
    def block_type(self) -> str:
        return "sdxl/latent_init"
