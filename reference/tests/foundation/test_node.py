"""Tests for foundation.Node."""

import pytest

from yggdrasill.foundation.node import Node
from tests.foundation.helpers import IdentityBlock


def test_node_basic() -> None:
    block = IdentityBlock(block_id="b1")
    node = Node("n1", block)
    assert node.node_id == "n1"
    assert node.block is block
    assert node.get_input_ports() == block.get_input_ports()
    assert node.get_output_ports() == block.get_output_ports()


def test_node_empty_id_raises() -> None:
    block = IdentityBlock(block_id="b1")
    with pytest.raises(ValueError, match="non-empty"):
        Node("", block)
    with pytest.raises(ValueError, match="non-empty"):
        Node("  ", block)
