import pytest
from yggdrasill.engine.executor import run, ValidationError
from yggdrasill.engine.planner import clear_plan_cache
from tests.engine.helpers import make_chain, make_cycle
from yggdrasill.engine.structure import Hypergraph
from tests.foundation.helpers import IdentityTaskNode


class TestExecutorChain:
    def test_chain_three(self):
        clear_plan_cache()
        h = make_chain("A", "B", "C")
        out = run(h, {"x": 42})
        assert out["y"] == 42

    def test_chain_two(self):
        clear_plan_cache()
        h = make_chain("A", "B")
        out = run(h, {"x": "hello"})
        assert out["y"] == "hello"

    def test_single_node(self):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("X", IdentityTaskNode(node_id="X"))
        h.expose_input("X", "in", "x")
        h.expose_output("X", "out", "y")
        out = run(h, {"x": 99})
        assert out["y"] == 99


class TestExecutorCycle:
    def test_cycle_two_steps(self):
        clear_plan_cache()
        h = make_cycle("A", "B")
        out = run(h, {"x": 1}, num_loop_steps=2, validate_before=False)
        assert out["y"] == 1  # identity pass-through on each iteration


class TestExecutorDryRun:
    def test_dry_run(self):
        clear_plan_cache()
        h = make_chain("A", "B")
        out = run(h, {"x": 42}, dry_run=True)
        assert "y" in out
        assert out["y"] is None


class TestExecutorValidation:
    def test_invalid_graph_raises(self):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        # required input not covered
        with pytest.raises(ValidationError):
            run(h, {})

    def test_skip_validation(self):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.expose_input("A", "in", "x")
        h.expose_output("A", "out", "y")
        out = run(h, {"x": 5}, validate_before=False)
        assert out["y"] == 5


class TestExecutorCallbacks:
    def test_callbacks_called(self):
        clear_plan_cache()
        h = make_chain("A", "B")
        log = []
        run(h, {"x": 1}, callbacks=[lambda phase, info: log.append((phase, info["node_id"]))])
        assert ("before", "A") in log
        assert ("after", "A") in log
        assert ("before", "B") in log
        assert ("after", "B") in log
