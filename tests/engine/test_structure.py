import pytest
from yggdrasill.engine.edge import Edge
from yggdrasill.engine.structure import Hypergraph
from tests.foundation.helpers import IdentityTaskNode


class TestHypergraphNodes:
    def test_add_and_get_node(self):
        h = Hypergraph()
        n = IdentityTaskNode(node_id="A")
        h.add_node("A", n)
        assert h.get_node("A") is n
        assert "A" in h.node_ids

    def test_remove_node(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        h.add_edge(Edge("A", "out", "B", "in"))
        h.expose_input("A", "in", "x")
        h.remove_node("A")
        assert "A" not in h.node_ids
        assert h.get_edges() == []
        assert all(e.get("node_id") != "A" for e in h.get_input_spec())

    def test_empty_node_id_raises(self):
        h = Hypergraph()
        with pytest.raises(ValueError):
            h.add_node("", IdentityTaskNode(node_id="A"))


class TestHypergraphEdges:
    def test_add_edge(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        h.add_edge(Edge("A", "out", "B", "in"))
        assert len(h.get_edges()) == 1

    def test_add_edge_idempotent(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        e = Edge("A", "out", "B", "in")
        h.add_edge(e)
        v1 = h.execution_version
        h.add_edge(e)
        assert len(h.get_edges()) == 1
        assert h.execution_version == v1  # no increment on duplicate

    def test_add_edge_unknown_node(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        with pytest.raises(ValueError, match="Target node"):
            h.add_edge(Edge("A", "out", "Z", "in"))

    def test_add_edge_unknown_port(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        with pytest.raises(ValueError, match="Port"):
            h.add_edge(Edge("A", "nonexistent", "B", "in"))

    def test_get_edges_in_out(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        h.add_edge(Edge("A", "out", "B", "in"))
        assert len(h.get_edges_out("A")) == 1
        assert len(h.get_edges_in("B")) == 1
        assert len(h.get_edges_in("A")) == 0


class TestHypergraphExposed:
    def test_expose_input_output(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.expose_input("A", "in", "x")
        h.expose_output("A", "out", "y")
        spec_in = h.get_input_spec()
        spec_out = h.get_output_spec()
        assert len(spec_in) == 1
        assert spec_in[0]["name"] == "x"
        assert len(spec_out) == 1
        assert spec_out[0]["name"] == "y"

    def test_expose_input_no_dup(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.expose_input("A", "in", "x")
        h.expose_input("A", "in", "x")
        assert len(h.get_input_spec()) == 1

    def test_include_dtype(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.expose_input("A", "in", "x")
        spec = h.get_input_spec(include_dtype=True)
        assert "dtype" in spec[0]


class TestExecutionVersion:
    def test_increments(self):
        h = Hypergraph()
        v0 = h.execution_version
        h.add_node("A", IdentityTaskNode(node_id="A"))
        assert h.execution_version > v0
        v1 = h.execution_version
        h.add_node("B", IdentityTaskNode(node_id="B"))
        assert h.execution_version > v1
        v2 = h.execution_version
        h.add_edge(Edge("A", "out", "B", "in"))
        assert h.execution_version > v2
