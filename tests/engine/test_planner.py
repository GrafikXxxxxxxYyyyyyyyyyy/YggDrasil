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


class TestPlannerEdgeCases:
    def test_empty_graph(self):
        clear_plan_cache()
        h = Hypergraph()
        plan = build_plan(h)
        assert plan == []

    def test_disconnected_components(self):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        h.expose_input("A", "in", "xa")
        h.expose_output("A", "out", "ya")
        h.expose_input("B", "in", "xb")
        h.expose_output("B", "out", "yb")
        plan = build_plan(h)
        plan_ids = set()
        for step_type, step_data in plan:
            if step_type == "node":
                plan_ids.add(step_data)
        assert "A" in plan_ids
        assert "B" in plan_ids


class TestPlannerLongChain:
    def test_five_node_dag_chain(self):
        clear_plan_cache()
        h = make_chain("A", "B", "C", "D", "E")
        plan = build_plan(h)
        assert len(plan) == 5
        ids = [step[1] for step in plan]
        assert ids == ["A", "B", "C", "D", "E"]
        assert all(s[0] == "node" for s in plan)


class TestPlannerMixedTopology:
    def test_dag_plus_cycle(self):
        """DAG prefix feeding into a cycle: A -> B <-> C."""
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        h.add_node("C", IdentityTaskNode(node_id="C"))
        h.add_edge(Edge("A", "out", "B", "in"))
        h.add_edge(Edge("B", "out", "C", "in"))
        h.add_edge(Edge("C", "out", "B", "in"))
        h.expose_input("A", "in", "x")
        h.expose_output("C", "out", "y")
        h.metadata = {"num_loop_steps": 2}

        plan = build_plan(h)
        types = [s[0] for s in plan]
        assert "node" in types
        assert "cycle" in types
        node_step_ids = [s[1] for s in plan if s[0] == "node"]
        assert "A" in node_step_ids


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

    def test_cache_eviction(self):
        """Fill the cache beyond _MAX_CACHE_SIZE to trigger eviction."""
        from yggdrasill.engine import planner
        clear_plan_cache()
        old_max = planner._MAX_CACHE_SIZE
        planner._MAX_CACHE_SIZE = 2
        try:
            h1 = make_chain("A", "B")
            h2 = make_chain("C", "D")
            h3 = make_chain("E", "F")
            build_plan(h1)
            build_plan(h2)
            build_plan(h3)
            assert len(planner._plan_cache) <= 2
        finally:
            planner._MAX_CACHE_SIZE = old_max
            clear_plan_cache()


class TestPlannerCacheTypeCollision:
    """B-1: Hypergraph and Workflow with same _instance_id must not collide."""

    def test_hypergraph_and_workflow_same_id_no_collision(self):
        from yggdrasill.workflow.workflow import Workflow
        from yggdrasill.foundation.registry import BlockRegistry

        clear_plan_cache()
        reg = BlockRegistry()
        reg.register("test/identity_task", IdentityTaskNode)

        h = Hypergraph.from_config({
            "graph_id": "h1",
            "nodes": [{"node_id": "N", "block_type": "test/identity_task"}],
            "edges": [],
            "exposed_inputs": [{"node_id": "N", "port_name": "in", "name": "x"}],
            "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "y"}],
        }, registry=reg)

        w = Workflow()
        hg = Hypergraph.from_config({
            "graph_id": "inner",
            "nodes": [{"node_id": "N", "block_type": "test/identity_task"}],
            "edges": [],
            "exposed_inputs": [{"node_id": "N", "port_name": "in", "name": "in"}],
            "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "out"}],
        }, registry=reg)
        w.add_node("g1", hg)
        w.expose_input("g1", "in", "x")
        w.expose_output("g1", "out", "y")

        w._instance_id = h._instance_id
        w._execution_version = h._execution_version

        plan_h = build_plan(h)
        plan_w = build_plan(w)
        assert plan_h[0][1] == "N"
        assert plan_w[0][1] == "g1"


class TestPlannerDiamondDAG:
    """Diamond topology: A -> B, A -> C, B -> D, C -> D."""

    def test_diamond_order(self):
        clear_plan_cache()
        h = Hypergraph()
        for nid in ("A", "B", "C", "D"):
            h.add_node(nid, IdentityTaskNode(node_id=nid))
        h.add_edge(Edge("A", "out", "B", "in"))
        h.add_edge(Edge("A", "out", "C", "in"))
        h.add_edge(Edge("B", "out", "D", "in"))
        h.add_edge(Edge("C", "out", "D", "in"))
        h.expose_input("A", "in", "x")
        h.expose_output("D", "out", "y")
        plan = build_plan(h)
        ids = [s[1] for s in plan]
        assert ids.index("A") < ids.index("B")
        assert ids.index("A") < ids.index("C")
        assert ids.index("B") < ids.index("D")
        assert ids.index("C") < ids.index("D")
