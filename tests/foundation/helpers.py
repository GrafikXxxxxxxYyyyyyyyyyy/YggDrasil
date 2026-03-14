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
    """Block that owns a child AddBlock, for testing nested state_dict."""

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

    def state_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for name, sub in self.get_sub_blocks().items():
            for k, v in sub.state_dict().items():
                result[f"{name}.{k}"] = v
        return result

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        child_state = {
            k.split(".", 1)[1]: v for k, v in state.items() if k.startswith("child.")
        }
        self.child.load_state_dict(child_state, strict=strict)


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
