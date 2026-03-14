from yggdrasill.engine.validator import validate
from tests.engine.helpers import make_chain, make_cycle
from yggdrasill.engine.edge import Edge
from yggdrasill.engine.structure import Hypergraph
from tests.foundation.helpers import IdentityTaskNode


class TestValidatorValid:
    def test_chain_valid(self):
        h = make_chain("A", "B", "C")
        r = validate(h)
        assert r.valid

    def test_cycle_valid(self):
        h = make_cycle()
        r = validate(h)
        assert r.valid


class TestValidatorErrors:
    def test_missing_port(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        h._edges.append(Edge("A", "nonexistent", "B", "in"))
        r = validate(h)
        assert not r.valid
        assert any("nonexistent" in e for e in r.errors)

    def test_required_input_uncovered(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        r = validate(h)
        assert not r.valid
        assert any("Required input" in e for e in r.errors)
