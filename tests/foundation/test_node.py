import pytest

from yggdrasill.foundation.port import PortDirection
from tests.foundation.helpers import IdentityTaskNode


class TestNodeCreation:
    def test_node_id(self):
        n = IdentityTaskNode(node_id="A")
        assert n.node_id == "A"

    def test_empty_node_id_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            IdentityTaskNode(node_id="")

    def test_whitespace_node_id_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            IdentityTaskNode(node_id="   ")

    def test_node_id_stripped(self):
        n = IdentityTaskNode(node_id="  B  ")
        assert n.node_id == "B"


class TestNodePorts:
    def test_declare_ports(self):
        n = IdentityTaskNode(node_id="A")
        ports = n.declare_ports()
        assert len(ports) == 2
        names = {p.name for p in ports}
        assert names == {"in", "out"}

    def test_get_input_ports(self):
        n = IdentityTaskNode(node_id="A")
        ins = n.get_input_ports()
        assert len(ins) == 1
        assert ins[0].direction == PortDirection.IN

    def test_get_output_ports(self):
        n = IdentityTaskNode(node_id="A")
        outs = n.get_output_ports()
        assert len(outs) == 1
        assert outs[0].direction == PortDirection.OUT

    def test_get_port(self):
        n = IdentityTaskNode(node_id="A")
        assert n.get_port("in") is not None
        assert n.get_port("out") is not None
        assert n.get_port("nonexistent") is None


class TestNodeRun:
    def test_run_delegates_to_forward(self):
        n = IdentityTaskNode(node_id="A")
        result = n.run({"in": 42})
        assert result == {"out": 42}
        assert n.run({"in": 42}) == n.forward({"in": 42})

    def test_block_and_node_are_same_object(self):
        n = IdentityTaskNode(node_id="A", block_id="b1")
        assert hasattr(n, "node_id")
        assert hasattr(n, "block_id")
        assert hasattr(n, "forward")
        assert hasattr(n, "declare_ports")
        assert n.node_id == "A"
        assert n.block_id == "b1"


class TestNodeRunWithoutForward:
    def test_run_without_dual_inheritance_raises_type_error(self):
        """A node that only inherits AbstractGraphNode (no forward) should raise TypeError."""
        from yggdrasill.foundation.node import AbstractGraphNode
        from yggdrasill.foundation.port import Port, PortDirection

        class PureNode(AbstractGraphNode):
            def declare_ports(self):
                return [Port("x", PortDirection.IN)]

        n = PureNode(node_id="test")
        with pytest.raises(TypeError, match="forward"):
            n.run({"x": 1})


class TestNodeRepr:
    def test_repr_contains_node_id(self):
        n = IdentityTaskNode(node_id="X")
        assert "X" in repr(n)

    def test_repr_contains_block_info(self):
        n = IdentityTaskNode(node_id="X", block_id="b1")
        r = repr(n)
        assert "block_type=" in r
        assert "block_id=" in r
        assert "b1" in r
