"""Concrete stubs for testing the foundation layer."""
from __future__ import annotations

from typing import Any, Dict, List

from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.foundation.block import AbstractBaseBlock
from yggdrasill.foundation.node import AbstractGraphNode


class IdentityBlock(AbstractBaseBlock):
    """forward: y = x."""

    @property
    def block_type(self) -> str:
        return "test/identity"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"y": inputs["x"]}


class AddBlock(AbstractBaseBlock):
    """forward: out = a + b + offset.  offset lives in config *and* state."""

    def __init__(self, block_id: str | None = None, *, config: dict | None = None) -> None:
        super().__init__(block_id=block_id, config=config)
        self.offset: float = (config or {}).get("offset", 0)

    @property
    def block_type(self) -> str:
        return "test/add"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": inputs["a"] + inputs["b"] + self.offset}

    def state_dict(self) -> Dict[str, Any]:
        return {"offset": self.offset}

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        if strict:
            expected = {"offset"}
            extra = set(state.keys()) - expected
            missing = expected - set(state.keys())
            if extra or missing:
                raise KeyError(f"state_dict mismatch: extra={extra}, missing={missing}")
        self.offset = state.get("offset", self.offset)


class BlockWithSub(AbstractBaseBlock):
    """Block that owns a child AddBlock, for testing nested state_dict.

    Uses the base-class auto-aggregation via get_sub_blocks().
    """

    def __init__(self, block_id: str | None = None, *, config: dict | None = None) -> None:
        super().__init__(block_id=block_id, config=config)
        self.child = AddBlock(config={"offset": (config or {}).get("child_offset", 0)})

    @property
    def block_type(self) -> str:
        return "test/with_sub"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self.child.forward(inputs)

    def get_sub_blocks(self) -> Dict[str, AbstractBaseBlock]:
        return {"child": self.child}


class IdentityTaskNode(AbstractBaseBlock, AbstractGraphNode):
    """One object, two origins.  For testing the Node interface."""

    def __init__(
        self,
        node_id: str,
        block_id: str | None = None,
        *,
        config: dict | None = None,
    ) -> None:
        AbstractBaseBlock.__init__(self, block_id=block_id, config=config)
        AbstractGraphNode.__init__(self, node_id=node_id)

    @property
    def block_type(self) -> str:
        return "test/identity_task"

    def declare_ports(self) -> List[Port]:
        return [
            Port("in", PortDirection.IN, PortType.ANY),
            Port("out", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": inputs["in"]}


class OptionalPortTaskNode(AbstractBaseBlock, AbstractGraphNode):
    """Task-node with one required and one optional input port."""

    def __init__(
        self,
        node_id: str,
        block_id: str | None = None,
        *,
        config: dict | None = None,
    ) -> None:
        AbstractBaseBlock.__init__(self, block_id=block_id, config=config)
        AbstractGraphNode.__init__(self, node_id=node_id)

    @property
    def block_type(self) -> str:
        return "test/optional_port"

    def declare_ports(self) -> List[Port]:
        return [
            Port("required_in", PortDirection.IN, PortType.ANY),
            Port("optional_in", PortDirection.IN, PortType.ANY, optional=True),
            Port("out", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        base = inputs["required_in"]
        extra = inputs.get("optional_in")
        if extra is not None:
            return {"out": f"{base}+{extra}"}
        return {"out": base}


class AddTaskNode(AbstractBaseBlock, AbstractGraphNode):
    """Task-node that adds two inputs with an offset."""

    def __init__(
        self,
        node_id: str,
        block_id: str | None = None,
        *,
        config: dict | None = None,
    ) -> None:
        AbstractBaseBlock.__init__(self, block_id=block_id, config=config)
        AbstractGraphNode.__init__(self, node_id=node_id)
        self.offset: float = (config or {}).get("offset", 0)

    @property
    def block_type(self) -> str:
        return "test/add_task"

    def declare_ports(self) -> List[Port]:
        return [
            Port("a", PortDirection.IN, PortType.ANY),
            Port("b", PortDirection.IN, PortType.ANY),
            Port("out", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": inputs["a"] + inputs["b"] + self.offset}

    def state_dict(self) -> Dict[str, Any]:
        return {"offset": self.offset}

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        if strict:
            expected = {"offset"}
            extra = set(state.keys()) - expected
            missing = expected - set(state.keys())
            if extra or missing:
                raise KeyError(f"state_dict mismatch: extra={extra}, missing={missing}")
        self.offset = state.get("offset", self.offset)
