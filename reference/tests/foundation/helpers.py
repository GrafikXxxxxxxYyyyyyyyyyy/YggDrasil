"""Concrete block and helpers for foundation tests."""

from typing import Any, Dict, List

from yggdrasill.foundation.block import AbstractBaseBlock
from yggdrasill.foundation.port import Port, PortDirection, PortType


class AddBlock(AbstractBaseBlock):
    """Block with in: a, b; out: out; out = a + b. Holds optional scalar state."""

    def __init__(self, block_id: str | None = None, *, config: dict | None = None) -> None:
        super().__init__(block_id=block_id, config=config)
        self.offset = (config or {}).get("offset", 0)

    @property
    def block_type(self) -> str:
        return "add"

    def declare_ports(self) -> List[Port]:
        return [
            Port("a", PortDirection.IN, dtype=PortType.ANY),
            Port("b", PortDirection.IN, dtype=PortType.ANY),
            Port("out", PortDirection.OUT, dtype=PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        a = inputs.get("a", 0)
        b = inputs.get("b", 0)
        return {"out": a + b + self.offset}

    def state_dict(self) -> Dict[str, Any]:
        return {"offset": self.offset}

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        if "offset" in state:
            self.offset = state["offset"]
        if strict and set(state) - {"offset"}:
            raise KeyError(f"Unexpected keys: {set(state) - {'offset'}}")


class IdentityBlock(AbstractBaseBlock):
    """Single in, single out; passes through. For graph tests."""

    def __init__(self, block_id: str | None = None, *, config: dict | None = None) -> None:
        super().__init__(block_id=block_id, config=config)

    @property
    def block_type(self) -> str:
        return "identity"

    def declare_ports(self) -> List[Port]:
        return [
            Port("x", PortDirection.IN, dtype=PortType.ANY),
            Port("y", PortDirection.OUT, dtype=PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"y": inputs.get("x")}
