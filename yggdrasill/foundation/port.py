from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union


class PortDirection(Enum):
    IN = "in"
    OUT = "out"


class PortType(Enum):
    TENSOR = "tensor"
    DICT = "dict"
    ANY = "any"
    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"


class PortAggregation(Enum):
    SINGLE = "single"
    CONCAT = "concat"
    SUM = "sum"
    FIRST = "first"
    DICT = "dict"


@dataclass(frozen=True)
class Port:
    """Describes a single input or output of a node in the hypergraph."""

    name: str
    direction: PortDirection
    dtype: Union[PortType, str] = PortType.ANY
    optional: bool = False
    aggregation: PortAggregation = PortAggregation.SINGLE

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Port name must be a non-empty string")
        if self.direction == PortDirection.OUT and self.aggregation != PortAggregation.SINGLE:
            raise ValueError(
                f"Output port '{self.name}' must have aggregation=SINGLE, got {self.aggregation}"
            )

    @property
    def is_input(self) -> bool:
        return self.direction == PortDirection.IN

    @property
    def is_output(self) -> bool:
        return self.direction == PortDirection.OUT

    def compatible_with(self, other: Port) -> bool:
        """Check if this (output) port can connect to other (input) port."""
        if self.direction != PortDirection.OUT or other.direction != PortDirection.IN:
            return False
        if self.dtype == PortType.ANY or other.dtype == PortType.ANY:
            return True
        return self.dtype == other.dtype
