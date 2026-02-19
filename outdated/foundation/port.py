"""
Ports: interface between blocks and graph edges.

Canon: WorldGenerator_2.0/Abstract_Block_And_Node.md §2.4, TODO_01 §2.
- Name, direction (in/out), type, optional, multiple/aggregation.
- Compatibility when connecting source port → target port.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PortDirection(str, Enum):
    IN = "in"
    OUT = "out"


class PortAggregation(str, Enum):
    """How to combine multiple incoming edges on one input port."""

    SINGLE = "single"  # exactly one edge allowed
    CONCAT = "concat"  # concatenate (e.g. tensors)
    SUM = "sum"
    FIRST = "first"   # take first value
    DICT = "dict"     # merge as dict by edge key


class PortType(str, Enum):
    """Data type for compatibility check."""

    TENSOR = "tensor"
    DICT = "dict"
    ANY = "any"
    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass(frozen=True)
class Port:
    """
    Single port: name, direction, type, optional, aggregation policy.

    Used in block port declarations and in graph edges (node_id, port_name).
    """

    name: str
    direction: PortDirection
    dtype: PortType = PortType.ANY
    optional: bool = False
    aggregation: PortAggregation = PortAggregation.SINGLE

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Port name must be non-empty")
        if self.direction == PortDirection.IN and self.aggregation != PortAggregation.SINGLE:
            # multiple edges allowed only when aggregation is set
            pass  # OK
        if self.direction == PortDirection.OUT and self.aggregation != PortAggregation.SINGLE:
            raise ValueError("Output ports use SINGLE aggregation only")

    @property
    def is_input(self) -> bool:
        return self.direction == PortDirection.IN

    @property
    def is_output(self) -> bool:
        return self.direction == PortDirection.OUT

    def compatible_with(self, other: Port) -> bool:
        """
        True if self (source) can connect to other (target).
        Source must be OUT, target must be IN; types compatible.
        """
        if self.direction != PortDirection.OUT or other.direction != PortDirection.IN:
            return False
        if self.dtype == PortType.ANY or other.dtype == PortType.ANY:
            return True
        return self.dtype == other.dtype
