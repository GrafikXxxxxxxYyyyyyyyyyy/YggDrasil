import pytest
from yggdrasill.engine.edge import Edge


class TestEdge:
    def test_creation(self):
        e = Edge("A", "out", "B", "in")
        assert e.source_node == "A"
        assert e.source_port == "out"
        assert e.target_node == "B"
        assert e.target_port == "in"

    def test_equality(self):
        e1 = Edge("A", "out", "B", "in")
        e2 = Edge("A", "out", "B", "in")
        assert e1 == e2

    def test_inequality(self):
        e1 = Edge("A", "out", "B", "in")
        e2 = Edge("A", "out", "C", "in")
        assert e1 != e2

    def test_hash(self):
        e1 = Edge("A", "out", "B", "in")
        e2 = Edge("A", "out", "B", "in")
        assert hash(e1) == hash(e2)
        assert len({e1, e2}) == 1

    def test_frozen(self):
        e = Edge("A", "out", "B", "in")
        with pytest.raises(AttributeError):
            e.source_node = "X"  # type: ignore[misc]

    def test_empty_source_node_raises(self):
        with pytest.raises(ValueError):
            Edge("", "out", "B", "in")

    def test_empty_target_port_raises(self):
        with pytest.raises(ValueError):
            Edge("A", "out", "B", "")

    def test_non_string_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a string"):
            Edge(123, "out", "B", "in")  # type: ignore[arg-type]

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a string"):
            Edge("A", None, "B", "in")  # type: ignore[arg-type]

    def test_whitespace_stripped(self):
        e = Edge("  A  ", "out", "B", "in")
        assert e.source_node == "A"
