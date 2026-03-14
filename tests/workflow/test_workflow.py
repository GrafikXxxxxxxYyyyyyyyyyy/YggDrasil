import pytest
from yggdrasill.engine.edge import Edge
from yggdrasill.engine.planner import clear_plan_cache
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.foundation.registry import BlockRegistry
from yggdrasill.workflow.workflow import Workflow
from tests.foundation.helpers import AddTaskNode, IdentityTaskNode

import yggdrasill.task_nodes.stubs  # noqa: F401


@pytest.fixture()
def registry():
    r = BlockRegistry()
    r.register("test/identity_task", IdentityTaskNode)
    r.register("test/add_task", AddTaskNode)
    return r


def _make_identity_hg(graph_id: str, registry: BlockRegistry) -> Hypergraph:
    """A single-node identity hypergraph with in->out."""
    return Hypergraph.from_config({
        "graph_id": graph_id,
        "nodes": [{"node_id": "N", "block_type": "test/identity_task"}],
        "edges": [],
        "exposed_inputs": [{"node_id": "N", "port_name": "in", "name": "in"}],
        "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "out"}],
    }, registry=registry)


def _make_add_hg(graph_id: str, offset: float, registry: BlockRegistry) -> Hypergraph:
    """A single-node add hypergraph with a+b->out."""
    return Hypergraph.from_config({
        "graph_id": graph_id,
        "nodes": [{"node_id": "N", "block_type": "test/add_task", "config": {"offset": offset}}],
        "edges": [],
        "exposed_inputs": [
            {"node_id": "N", "port_name": "a", "name": "a"},
            {"node_id": "N", "port_name": "b", "name": "b"},
        ],
        "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "out"}],
    }, registry=registry)


class TestWorkflowBasic:
    def test_add_and_get_node(self, registry):
        w = Workflow()
        hg = _make_identity_hg("hg1", registry)
        w.add_node("step1", hg)
        assert "step1" in w.node_ids
        assert w.get_node("step1") is hg

    def test_remove_node(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.remove_node("s1")
        assert "s1" not in w.node_ids


class TestWorkflowRun:
    def test_chain_of_two_hypergraphs(self, registry):
        clear_plan_cache()
        w = Workflow(workflow_id="w_chain")
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.add_node("s2", _make_identity_hg("hg2", registry))
        w.add_edge(Edge("s1", "out", "s2", "in"))
        w.expose_input("s1", "in", "x")
        w.expose_output("s2", "out", "y")

        result = w.run({"x": 42}, validate_before=False)
        assert result["y"] == 42

    def test_chain_of_three(self, registry):
        clear_plan_cache()
        w = Workflow()
        w.add_node("a", _make_identity_hg("hg_a", registry))
        w.add_node("b", _make_identity_hg("hg_b", registry))
        w.add_node("c", _make_identity_hg("hg_c", registry))
        w.add_edge(Edge("a", "out", "b", "in"))
        w.add_edge(Edge("b", "out", "c", "in"))
        w.expose_input("a", "in", "x")
        w.expose_output("c", "out", "y")

        result = w.run({"x": "hello"}, validate_before=False)
        assert result["y"] == "hello"


class TestWorkflowConfig:
    def test_to_config(self, registry):
        w = Workflow(workflow_id="test_wf")
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.expose_input("s1", "in", "x")
        w.expose_output("s1", "out", "y")
        cfg = w.to_config()
        assert cfg["schema_version"] == "1.0"
        assert cfg["workflow_id"] == "test_wf"
        assert len(cfg["nodes"]) == 1

    def test_from_config_roundtrip(self, registry):
        clear_plan_cache()
        w1 = Workflow(workflow_id="rt_wf")
        w1.add_node("s1", _make_identity_hg("hg1", registry))
        w1.add_node("s2", _make_identity_hg("hg2", registry))
        w1.add_edge(Edge("s1", "out", "s2", "in"))
        w1.expose_input("s1", "in", "x")
        w1.expose_output("s2", "out", "y")

        cfg = w1.to_config()
        w2 = Workflow.from_config(cfg, registry=registry)
        assert w2.graph_id == "rt_wf"
        assert w2.node_ids == w1.node_ids

        result = w2.run({"x": 7}, validate_before=False)
        assert result["y"] == 7


class TestWorkflowStateDict:
    def test_state_dict(self, registry):
        w = Workflow()
        w.add_node("s1", _make_add_hg("hg_add", 99, registry))
        sd = w.state_dict()
        assert "s1" in sd
        assert sd["s1"]["N"]["offset"] == 99

    def test_load_state_dict(self, registry):
        clear_plan_cache()
        w = Workflow()
        w.add_node("s1", _make_add_hg("hg_add", 0, registry))
        w.expose_input("s1", "a", "a")
        w.expose_input("s1", "b", "b")
        w.expose_output("s1", "out", "out")
        w.load_state_dict({"s1": {"N": {"offset": 50}}})
        result = w.run({"a": 0, "b": 0}, validate_before=False)
        assert result["out"] == 50


class TestWorkflowSaveLoad:
    def test_save_load_roundtrip(self, tmp_path, registry):
        clear_plan_cache()
        w1 = Workflow(workflow_id="persist_wf")
        w1.add_node("s1", _make_identity_hg("hg1", registry))
        w1.add_node("s2", _make_identity_hg("hg2", registry))
        w1.add_edge(Edge("s1", "out", "s2", "in"))
        w1.expose_input("s1", "in", "x")
        w1.expose_output("s2", "out", "y")

        w1.save(tmp_path / "wf")
        w2 = Workflow.load(tmp_path / "wf", registry=registry)

        assert w2.graph_id == "persist_wf"
        result = w2.run({"x": "data"}, validate_before=False)
        assert result["y"] == "data"

    def test_save_load_with_state(self, tmp_path, registry):
        clear_plan_cache()
        w1 = Workflow()
        w1.add_node("s1", _make_add_hg("hg_add", 42, registry))
        w1.expose_input("s1", "a", "a")
        w1.expose_input("s1", "b", "b")
        w1.expose_output("s1", "out", "out")
        out1 = w1.run({"a": 1, "b": 2}, validate_before=False)

        w1.save(tmp_path / "wf_state")
        w2 = Workflow.load(tmp_path / "wf_state", registry=registry)
        out2 = w2.run({"a": 1, "b": 2}, validate_before=False)
        assert out1 == out2


class TestWorkflowCycle:
    def test_cycle_between_hypergraphs(self, registry):
        clear_plan_cache()
        w = Workflow()
        w.add_node("a", _make_identity_hg("cyc_a", registry))
        w.add_node("b", _make_identity_hg("cyc_b", registry))
        w.add_edge(Edge("a", "out", "b", "in"))
        w.add_edge(Edge("b", "out", "a", "in"))
        w.expose_input("a", "in", "x")
        w.expose_output("b", "out", "y")
        w.metadata = {"num_loop_steps": 3}

        result = w.run({"x": "loop"}, validate_before=False, num_loop_steps=3)
        assert result["y"] == "loop"
