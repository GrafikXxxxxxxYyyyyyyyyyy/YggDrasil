"""
Concrete stub blocks for each role: identity/passthrough for tests and from_template.

Canon: WorldGenerator_2.0/TODO_02. block_type = "role/identity" so role_from_block_type works.
"""

from __future__ import annotations

from typing import Any, Dict

from yggdrasill.foundation.registry import BlockRegistry, register_block
from yggdrasill.task_nodes.abstract import (
    AbstractAdapter,
    AbstractBackbone,
    AbstractCodec,
    AbstractConditioner,
    AbstractGuidance,
    AbstractSolver,
    AbstractTokenizer,
)


@register_block("backbone/identity")
class IdentityBackbone(AbstractBackbone):
    """Passthrough: pred = latent (or zero if no latent)."""

    @property
    def block_type(self) -> str:
        return "backbone/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"pred": inputs.get("latent")}


@register_block("solver/identity")
class IdentitySolver(AbstractSolver):
    """Passthrough: next_latent = latent, next_timestep = timestep."""

    @property
    def block_type(self) -> str:
        return "solver/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "next_latent": inputs.get("latent"),
            "next_timestep": inputs.get("timestep"),
        }


@register_block("codec/identity")
class IdentityCodec(AbstractCodec):
    """Passthrough: encode_latent = encode_image if present; decode_image = decode_latent."""

    @property
    def block_type(self) -> str:
        return "codec/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if "encode_image" in inputs and inputs["encode_image"] is not None:
            out["encode_latent"] = inputs["encode_image"]
        if "decode_latent" in inputs and inputs["decode_latent"] is not None:
            out["decode_image"] = inputs["decode_latent"]
        return out


@register_block("conditioner/identity")
class IdentityConditioner(AbstractConditioner):
    """Passthrough: embedding = input."""

    @property
    def block_type(self) -> str:
        return "conditioner/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"embedding": inputs.get("input")}


@register_block("tokenizer/identity")
class IdentityTokenizer(AbstractTokenizer):
    """Passthrough: token_ids = text (as-is)."""

    @property
    def block_type(self) -> str:
        return "tokenizer/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"token_ids": inputs.get("text")}


@register_block("adapter/identity")
class IdentityAdapter(AbstractAdapter):
    """Passthrough: adapted = condition."""

    @property
    def block_type(self) -> str:
        return "adapter/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"adapted": inputs.get("condition")}


@register_block("guidance/identity")
class IdentityGuidance(AbstractGuidance):
    """Passthrough: pred_guided = pred_cond."""

    @property
    def block_type(self) -> str:
        return "guidance/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"pred_guided": inputs.get("pred_cond")}


def register_task_node_stubs(registry: BlockRegistry | None = None) -> None:
    """Register all identity stubs in the given registry (default: global)."""
    reg = registry or BlockRegistry.global_registry()
    for block_type, cls in [
        ("backbone/identity", IdentityBackbone),
        ("solver/identity", IdentitySolver),
        ("codec/identity", IdentityCodec),
        ("conditioner/identity", IdentityConditioner),
        ("tokenizer/identity", IdentityTokenizer),
        ("adapter/identity", IdentityAdapter),
        ("guidance/identity", IdentityGuidance),
    ]:
        reg.register(block_type, cls)
