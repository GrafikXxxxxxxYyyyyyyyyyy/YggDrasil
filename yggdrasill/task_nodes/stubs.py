"""Identity stubs for each of the seven roles.

Registered as `<role>/identity` in the global BlockRegistry
so they can be used for testing and scaffolding.
"""
from __future__ import annotations

from typing import Any, Dict

from yggdrasill.foundation.registry import register_block
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
        return {"out": inputs.get("in")}


@register_block("injector/identity")
class IdentityInjector(AbstractInjector):
    @property
    def block_type(self) -> str:
        return "injector/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": inputs.get("condition")}


@register_block("conjector/identity")
class IdentityConjector(AbstractConjector):
    @property
    def block_type(self) -> str:
        return "conjector/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": inputs.get("in")}


@register_block("inner_module/identity")
class IdentityInnerModule(AbstractInnerModule):
    @property
    def block_type(self) -> str:
        return "inner_module/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": inputs.get("in")}


@register_block("outer_module/identity")
class IdentityOuterModule(AbstractOuterModule):
    @property
    def block_type(self) -> str:
        return "outer_module/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": inputs.get("in")}


@register_block("helper/identity")
class IdentityHelper(AbstractHelper):
    @property
    def block_type(self) -> str:
        return "helper/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": inputs.get("in")}


@register_block("converter/identity")
class IdentityConverter(AbstractConverter):
    @property
    def block_type(self) -> str:
        return "converter/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": inputs.get("in")}
