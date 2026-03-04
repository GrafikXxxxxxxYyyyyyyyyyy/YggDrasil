"""Tests for foundation.BlockRegistry."""

import pytest

from yggdrasill.foundation.registry import BlockRegistry, register_block
from yggdrasill.foundation.block import AbstractBaseBlock
from yggdrasill.foundation.port import Port, PortDirection, PortType
from tests.foundation.helpers import AddBlock, IdentityBlock


@pytest.fixture
def registry() -> BlockRegistry:
    r = BlockRegistry()
    r.register("add", AddBlock)
    r.register("identity", IdentityBlock)
    return r


def test_registry_build(registry: BlockRegistry) -> None:
    b = registry.build({"block_type": "add", "block_id": "a1", "offset": 2})
    assert b.block_type == "add"
    assert b.block_id == "a1"
    assert b.forward({"a": 1, "b": 0}) == {"out": 3}


def test_registry_build_with_type_key(registry: BlockRegistry) -> None:
    b = registry.build({"type": "identity", "block_id": "id1"})
    assert b.block_type == "identity"


def test_registry_unknown_type_raises(registry: BlockRegistry) -> None:
    with pytest.raises(KeyError, match="Unknown block_type"):
        registry.build({"block_type": "unknown"})


def test_registry_missing_type_raises(registry: BlockRegistry) -> None:
    with pytest.raises(KeyError, match="block_type"):
        registry.build({"block_id": "x"})


def test_register_block_decorator() -> None:
    r = BlockRegistry()

    @register_block("dummy", r)
    class DummyBlock(AbstractBaseBlock):
        def declare_ports(self):
            return [Port("x", PortDirection.IN), Port("y", PortDirection.OUT)]

        def forward(self, inputs):
            return {"y": inputs.get("x")}

    b = r.build({"block_type": "dummy", "block_id": "d1"})
    assert b.block_type == "DummyBlock"  # class name, not registered type
    assert b.forward({"x": 1}) == {"y": 1}
