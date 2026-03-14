"""Seven abstract task-node roles.

Each inherits both AbstractBaseBlock (material) and AbstractGraphNode (ideal),
following the dual-inheritance principle: one object, two origins.
Port contracts follow the canonical specification (canon 02).
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from yggdrasill.foundation.block import AbstractBaseBlock
from yggdrasill.foundation.node import AbstractGraphNode
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.roles import Role


class _TaskNodeBase(AbstractBaseBlock, AbstractGraphNode):
    """Shared boilerplate for all seven roles."""

    _role: Role

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        AbstractBaseBlock.__init__(self, block_id=block_id, config=config)
        AbstractGraphNode.__init__(self, node_id=node_id)

    @property
    def role(self) -> Role:
        return self._role


# ---------------------------------------------------------------------------
# 1. Backbone
# ---------------------------------------------------------------------------

class AbstractBackbone(_TaskNodeBase):
    """Primary transformation (e.g., UNet, Transformer decoder)."""

    _role = Role.BACKBONE

    @property
    def block_type(self) -> str:
        return "backbone"

    def declare_ports(self) -> List[Port]:
        return [
            Port("in", PortDirection.IN, PortType.ANY),
            Port("out", PortDirection.OUT, PortType.ANY),
        ]

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# 2. Injector
# ---------------------------------------------------------------------------

class AbstractInjector(_TaskNodeBase):
    """Injects conditioning signal into the backbone stream."""

    _role = Role.INJECTOR

    @property
    def block_type(self) -> str:
        return "injector"

    def declare_ports(self) -> List[Port]:
        return [
            Port("condition", PortDirection.IN, PortType.ANY),
            Port("out", PortDirection.OUT, PortType.ANY),
        ]

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# 3. Conjector
# ---------------------------------------------------------------------------

class AbstractConjector(_TaskNodeBase):
    """Merges/predicts outputs back into the stream."""

    _role = Role.CONJECTOR

    @property
    def block_type(self) -> str:
        return "conjector"

    def declare_ports(self) -> List[Port]:
        return [
            Port("in", PortDirection.IN, PortType.ANY),
            Port("out", PortDirection.OUT, PortType.ANY),
        ]

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# 4. Inner Module
# ---------------------------------------------------------------------------

class AbstractInnerModule(_TaskNodeBase):
    """Internal processing module (lives inside the task hypergraph)."""

    _role = Role.INNER_MODULE

    @property
    def block_type(self) -> str:
        return "inner_module"

    def declare_ports(self) -> List[Port]:
        return [
            Port("in", PortDirection.IN, PortType.ANY),
            Port("out", PortDirection.OUT, PortType.ANY),
        ]

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# 5. Outer Module
# ---------------------------------------------------------------------------

class AbstractOuterModule(_TaskNodeBase):
    """External processing module (interface between hypergraphs)."""

    _role = Role.OUTER_MODULE

    @property
    def block_type(self) -> str:
        return "outer_module"

    def declare_ports(self) -> List[Port]:
        return [
            Port("in", PortDirection.IN, PortType.ANY),
            Port("out", PortDirection.OUT, PortType.ANY),
        ]

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# 6. Helper
# ---------------------------------------------------------------------------

class AbstractHelper(_TaskNodeBase):
    """Utility node (logging, formatting, scheduling, etc.)."""

    _role = Role.HELPER

    @property
    def block_type(self) -> str:
        return "helper"

    def declare_ports(self) -> List[Port]:
        return [
            Port("in", PortDirection.IN, PortType.ANY, optional=True),
            Port("out", PortDirection.OUT, PortType.ANY),
        ]

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# 7. Converter
# ---------------------------------------------------------------------------

class AbstractConverter(_TaskNodeBase):
    """Converts data between representations (e.g., VAE encode/decode)."""

    _role = Role.CONVERTER

    @property
    def block_type(self) -> str:
        return "converter"

    def declare_ports(self) -> List[Port]:
        return [
            Port("in", PortDirection.IN, PortType.ANY),
            Port("out", PortDirection.OUT, PortType.ANY),
        ]

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...
