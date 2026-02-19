"""Tests for foundation.AbstractBaseBlock and concrete blocks."""

import pytest

from yggdrasill.foundation.block import AbstractBaseBlock
from yggdrasill.foundation.port import Port, PortDirection, PortType
from tests.foundation.helpers import AddBlock, IdentityBlock


def test_add_block_ports() -> None:
    b = AddBlock(block_id="add1")
    assert b.block_type == "add"
    assert b.block_id == "add1"
    ports = b.declare_ports()
    names = [p.name for p in ports]
    assert "a" in names and "b" in names and "out" in names
    assert len(b.get_input_ports()) == 2
    assert len(b.get_output_ports()) == 1


def test_add_block_forward() -> None:
    b = AddBlock(block_id="add1")
    out = b.forward({"a": 10, "b": 5})
    assert out == {"out": 15}


def test_add_block_config_offset() -> None:
    b = AddBlock(block_id="x", config={"offset": 3})
    out = b.forward({"a": 1, "b": 2})
    assert out == {"out": 6}


def test_add_block_state_dict_load() -> None:
    b = AddBlock(block_id="x", config={"offset": 7})
    sd = b.state_dict()
    assert sd == {"offset": 7}
    b2 = AddBlock(block_id="y", config={})
    b2.load_state_dict(sd)
    assert b2.forward({"a": 0, "b": 0}) == {"out": 7}


def test_identity_block() -> None:
    b = IdentityBlock(block_id="id1")
    assert b.forward({"x": 42}) == {"y": 42}


def test_block_train_eval_freeze() -> None:
    b = AddBlock(block_id="x")
    assert b.training is True
    b.eval()
    assert b.training is False
    b.train(True)
    assert b.training is True
    assert b.frozen is False
    b.freeze()
    assert b.frozen is True
    b.unfreeze()
    assert b.frozen is False


def test_block_run_alias() -> None:
    b = IdentityBlock(block_id="id1")
    assert b.run({"x": 1}) == b.forward({"x": 1})


def test_block_with_sub_blocks_state_dict_roundtrip() -> None:
    """Composite block: sub-blocks are serialized with prefix (TODO_01 §1.3, §7)."""
    from yggdrasill.foundation.block import AbstractBaseBlock

    class ParentBlock(AbstractBaseBlock):
        def __init__(self, block_id=None, *, config=None):
            super().__init__(block_id=block_id, config=config)
            self.child = AddBlock(block_id="c1", config={"offset": 5})

        @property
        def block_type(self):
            return "parent"

        def get_sub_blocks(self):
            return {"child": self.child}

        def declare_ports(self):
            return []

        def forward(self, inputs):
            return {}

    parent = ParentBlock(block_id="p1")
    sd = parent.state_dict()
    assert "child.offset" in sd
    assert sd["child.offset"] == 5
    parent.child.offset = 99
    parent.load_state_dict(sd, strict=True)
    assert parent.child.offset == 5
