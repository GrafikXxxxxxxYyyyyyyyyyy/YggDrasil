"""Identity stubs for each of the seven roles.

Registered as ``<role>/identity`` in the global BlockRegistry
so they can be used for testing and scaffolding.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from yggdrasill.foundation.registry import BlockRegistry, register_block
from yggdrasill.task_nodes.abstract import (
    AbstractBackbone,
    AbstractConjector,
    AbstractConverter,
    AbstractHelper,
    AbstractInjector,
    AbstractInnerModule,
    AbstractOuterModule,
)


@register_block("backbone/identity")
class IdentityBackbone(AbstractBackbone):
    @property
    def block_type(self) -> str:
        return "backbone/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"pred": inputs.get("latent")}


@register_block("injector/identity")
class IdentityInjector(AbstractInjector):
    @property
    def block_type(self) -> str:
        return "injector/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"adapted": inputs.get("condition")}


@register_block("conjector/identity")
class IdentityConjector(AbstractConjector):
    @property
    def block_type(self) -> str:
        return "conjector/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"condition": inputs.get("input")}


@register_block("inner_module/identity")
class IdentityInnerModule(AbstractInnerModule):
    @property
    def block_type(self) -> str:
        return "inner_module/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "next_latent": inputs.get("latent"),
            "next_timestep": inputs.get("timestep"),
        }


@register_block("outer_module/identity")
class IdentityOuterModule(AbstractOuterModule):
    @property
    def block_type(self) -> str:
        return "outer_module/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": inputs.get("input")}


@register_block("helper/identity")
class IdentityHelper(AbstractHelper):
    @property
    def block_type(self) -> str:
        return "helper/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": inputs.get("query")}


@register_block("converter/identity")
class IdentityConverter(AbstractConverter):
    @property
    def block_type(self) -> str:
        return "converter/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": inputs.get("input")}


def register_all_stubs(registry: Optional[BlockRegistry] = None) -> None:
    """Register all identity stubs in the given (or global) registry.

    Importing this module already registers stubs in the global registry
    via ``@register_block``.  Call this function to explicitly register
    them in a *different* registry instance.
    """
    reg = registry or BlockRegistry.global_registry()
    reg.register("backbone/identity", IdentityBackbone)
    reg.register("injector/identity", IdentityInjector)
    reg.register("conjector/identity", IdentityConjector)
    reg.register("inner_module/identity", IdentityInnerModule)
    reg.register("outer_module/identity", IdentityOuterModule)
    reg.register("helper/identity", IdentityHelper)
    reg.register("converter/identity", IdentityConverter)
