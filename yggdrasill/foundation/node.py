from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from yggdrasill.foundation.port import Port, PortDirection


class AbstractGraphNode(ABC):
    """Ideal origin: position and connections in the hypergraph.

    Provides node_id, port declarations, and the run() interface for the engine.
    Does not store a block -- used only via dual inheritance with AbstractBaseBlock
    in task-node classes, where run() delegates to self.forward().
    """

    def __init__(self, node_id: str) -> None:
        if not node_id or not node_id.strip():
            raise ValueError("node_id must be a non-empty string")
        self._node_id = node_id.strip()

    @property
    def node_id(self) -> str:
        return self._node_id

    @abstractmethod
    def declare_ports(self) -> List[Port]:
        """Declare the ports of this node (inputs and outputs for hyperedges)."""
        ...

    def get_input_ports(self) -> List[Port]:
        return [p for p in self.declare_ports() if p.direction == PortDirection.IN]

    def get_output_ports(self) -> List[Port]:
        return [p for p in self.declare_ports() if p.direction == PortDirection.OUT]

    def get_port(self, name: str) -> Port | None:
        """Look up a port by name, or return None."""
        for p in self.declare_ports():
            if p.name == name:
                return p
        return None

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Engine-facing entry point. Delegates to self.forward() (Block side)."""
        return self.forward(inputs)  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        return f"<{type(self).__name__} node_id={self._node_id!r}>"
