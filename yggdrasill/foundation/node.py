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
        self._ports_cache: List[Port] | None = None

    @property
    def node_id(self) -> str:
        return self._node_id

    @abstractmethod
    def declare_ports(self) -> List[Port]:
        """Declare the ports of this node (inputs and outputs for hyperedges)."""
        ...

    def _get_ports(self) -> List[Port]:
        if self._ports_cache is None:
            self._ports_cache = self.declare_ports()
        return self._ports_cache

    def invalidate_ports_cache(self) -> None:
        """Clear cached ports (call if declare_ports() result changes at runtime)."""
        self._ports_cache = None

    def get_input_ports(self) -> List[Port]:
        return [p for p in self._get_ports() if p.direction == PortDirection.IN]

    def get_output_ports(self) -> List[Port]:
        return [p for p in self._get_ports() if p.direction == PortDirection.OUT]

    def get_port(self, name: str) -> Port | None:
        """Look up a port by name, or return None."""
        for p in self._get_ports():
            if p.name == name:
                return p
        return None

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Engine-facing entry point. Delegates to self.forward() (Block side)."""
        if not hasattr(self, "forward"):
            raise TypeError(
                f"{type(self).__name__} does not implement forward(). "
                f"Dual-inherit from AbstractBaseBlock to get forward()."
            )
        return self.forward(inputs)  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        parts = [f"node_id={self._node_id!r}"]
        if hasattr(self, "block_type"):
            parts.append(f"block_type={self.block_type!r}")
        if hasattr(self, "block_id"):
            parts.append(f"block_id={self.block_id!r}")
        return f"<{type(self).__name__} {' '.join(parts)}>"
