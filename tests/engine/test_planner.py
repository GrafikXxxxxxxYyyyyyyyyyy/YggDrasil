from yggdrasill.engine.planner import build_plan, clear_plan_cache
from tests.engine.helpers import make_chain, make_cycle
from yggdrasill.engine.edge import Edge
from yggdrasill.engine.structure import Hypergraph
from tests.foundation.helpers import IdentityTaskNode


class TestPlannerDAG:
    def test_chain_order(self):
        clear_plan_cache()
        h = make_chain("A", "B", "C")
        plan = build_plan(h)
        assert len(plan) == 3
        ids = [step[1] for step in plan]
        assert ids == ["A", "B", "C"]
        assert all(s[0] == "node" for s in plan)

    def test_single_node(self):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("X", IdentityTaskNode(node_id="X"))
        h.expose_input("X", "in", "x")
        h.expose_output("X", "out", "y")
        plan = build_plan(h)
        assert plan == [("node", "X")]


class TestPlannerCycle:
    def test_two_node_cycle(self):
        clear_plan_cache()
        h = make_cycle("A", "B")
        plan = build_plan(h)
        assert len(plan) == 1
        assert plan[0][0] == "cycle"
        _, (rep, comp) = plan[0]
        assert comp == frozenset({"A", "B"})

    def test_self_loop(self):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("X", IdentityTaskNode(node_id="X"))
        h.add_edge(Edge("X", "out", "X", "in"))
        h.expose_input("X", "in", "x")
        h.expose_output("X", "out", "y")
        plan = build_plan(h)
        assert plan[0][0] == "cycle"


class TestPlannerCache:
    def test_cache_hit(self):
        clear_plan_cache()
        h = make_chain("A", "B")
        p1 = build_plan(h)
        p2 = build_plan(h)
        assert p1 is p2

    def test_cache_invalidation(self):
        clear_plan_cache()
        h = make_chain("A", "B")
        p1 = build_plan(h)
        h.add_node("C", IdentityTaskNode(node_id="C"))
        h.add_edge(Edge("B", "out", "C", "in"))
        p2 = build_plan(h)
        assert p1 is not p2
