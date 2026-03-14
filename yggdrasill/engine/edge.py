from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Edge:
    """A directed connection from one node's output port to another node's input port."""

    source_node: str
    source_port: str
    target_node: str
    target_port: str

    def __post_init__(self) -> None:
        for field in ("source_node", "source_port", "target_node", "target_port"):
            val = getattr(self, field)
            if not val or not val.strip():
                raise ValueError(f"Edge.{field} must be a non-empty string")
