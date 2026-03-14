import pytest
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
    return Hypergraph.from_config({
        "graph_id": graph_id,
        "nodes": [{"node_id": "N", "block_type": "test/identity_task"}],
        "edges": [],
        "exposed_inputs": [{"node_id": "N", "port_name": "in", "name": "in"}],
        "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "out"}],
    }, registry=registry)


def _make_add_hg(graph_id: str, offset: float, registry: BlockRegistry) -> Hypergraph:
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
        gid = w.add_node("step1", hg)
        assert gid == "step1"
        assert "step1" in w.node_ids
        assert w.get_node("step1") is hg

    def test_remove_node(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.remove_node("s1")
        assert "s1" not in w.node_ids

    def test_execution_version(self, registry):
        w = Workflow()
        v0 = w.execution_version
        w.add_node("s1", _make_identity_hg("hg1", registry))
        assert w.execution_version > v0


class TestWorkflowRun:
    def test_chain_of_two_hypergraphs(self, registry):
        clear_plan_cache()
        w = Workflow(workflow_id="w_chain")
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.add_node("s2", _make_identity_hg("hg2", registry))
        w.add_edge("s1", "out", "s2", "in")
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
        w.add_edge("a", "out", "b", "in")
        w.add_edge("b", "out", "c", "in")
        w.expose_input("a", "in", "x")
        w.expose_output("c", "out", "y")

        result = w.run({"x": "hello"}, validate_before=False)
        assert result["y"] == "hello"

    def test_single_hypergraph(self, registry):
        clear_plan_cache()
        w = Workflow()
        w.add_node("g", _make_identity_hg("hg", registry))
        w.expose_input("g", "in", "x")
        w.expose_output("g", "out", "y")
        result = w.run({"x": 99}, validate_before=False)
        assert result["y"] == 99


class TestWorkflowEdgeValidation:
    def test_add_edge_validates_ports(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.add_node("s2", _make_identity_hg("hg2", registry))
        with pytest.raises(ValueError, match="not in output spec"):
            w.add_edge("s1", "nonexistent", "s2", "in")

    def test_add_edge_unknown_graph(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        with pytest.raises(ValueError, match="not in workflow"):
            w.add_edge("s1", "out", "missing", "in")


class TestWorkflowRemoveEdge:
    def test_remove_edge(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.add_node("s2", _make_identity_hg("hg2", registry))
        w.add_edge("s1", "out", "s2", "in")
        assert len(w.get_edges()) == 1
        w.remove_edge("s1", "out", "s2", "in")
        assert len(w.get_edges()) == 0


class TestWorkflowConfig:
    def test_to_config(self, registry):
        w = Workflow(workflow_id="test_wf")
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.expose_input("s1", "in", "x")
        w.expose_output("s1", "out", "y")
        cfg = w.to_config()
        assert cfg["schema_version"] == "1.0"
        assert cfg["workflow_id"] == "test_wf"
        assert len(cfg["graphs"]) == 1
        assert cfg["graphs"][0]["graph_id"] == "s1"
        assert "config" in cfg["graphs"][0]

    def test_from_config_roundtrip(self, registry):
        clear_plan_cache()
        w1 = Workflow(workflow_id="rt_wf")
        w1.add_node("s1", _make_identity_hg("hg1", registry))
        w1.add_node("s2", _make_identity_hg("hg2", registry))
        w1.add_edge("s1", "out", "s2", "in")
        w1.expose_input("s1", "in", "x")
        w1.expose_output("s2", "out", "y")

        cfg = w1.to_config()
        w2 = Workflow.from_config(cfg, registry=registry)
        assert w2.graph_id == "rt_wf"
        assert w2.node_ids == w1.node_ids

        result = w2.run({"x": 7}, validate_before=False)
        assert result["y"] == 7

    def test_workflow_kind(self, registry):
        w = Workflow(workflow_id="kind_wf")
        w.workflow_kind = "chain"
        w.add_node("s1", _make_identity_hg("hg1", registry))
        cfg = w.to_config()
        assert cfg["workflow_kind"] == "chain"
        w2 = Workflow.from_config(cfg, registry=registry)
        assert w2.workflow_kind == "chain"


class TestWorkflowStateDict:
    def test_state_dict(self, registry):
        w = Workflow()
        w.add_node("s1", _make_add_hg("hg_add", 99, registry))
        sd = w.state_dict()
        assert "s1" in sd
        assert sd["s1"]["N"]["offset"] == 99

    def test_state_dict_empty_graph(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        sd = w.state_dict()
        assert "s1" in sd
        assert sd["s1"] == {}

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

    def test_load_state_dict_strict(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        with pytest.raises(KeyError, match="not in workflow"):
            w.load_state_dict({"unknown": {}}, strict=True)


class TestWorkflowTrainable:
    def test_set_trainable(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.set_trainable("s1", False)
        params = list(w.trainable_parameters())
        assert params == []

    def test_set_trainable_unknown(self, registry):
        w = Workflow()
        with pytest.raises(ValueError, match="not in workflow"):
            w.set_trainable("missing", False)


class TestWorkflowTo:
    def test_to_device(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        result = w.to("cpu")
        assert result is w


class TestWorkflowInferExposed:
    def test_infer_exposed_ports(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.add_node("s2", _make_identity_hg("hg2", registry))
        w.add_edge("s1", "out", "s2", "in")
        w.infer_exposed_ports()
        input_ports = {e["port_name"] for e in w.get_input_spec()}
        output_ports = {e["port_name"] for e in w.get_output_spec()}
        assert "in" in input_ports
        assert "out" in output_ports


class TestWorkflowSaveLoad:
    def test_save_load_roundtrip(self, tmp_path, registry):
        clear_plan_cache()
        w1 = Workflow(workflow_id="persist_wf")
        w1.add_node("s1", _make_identity_hg("hg1", registry))
        w1.add_node("s2", _make_identity_hg("hg2", registry))
        w1.add_edge("s1", "out", "s2", "in")
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

    def test_save_config_only(self, tmp_path, registry):
        w = Workflow(workflow_id="cfg_only")
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.expose_input("s1", "in", "x")
        w.expose_output("s1", "out", "y")
        w.save_config(tmp_path / "wf_cfg")
        assert (tmp_path / "wf_cfg" / "config.json").exists()

    def test_load_from_checkpoint(self, tmp_path, registry):
        clear_plan_cache()
        w1 = Workflow()
        w1.add_node("s1", _make_add_hg("hg_add", 77, registry))
        w1.expose_input("s1", "a", "a")
        w1.expose_input("s1", "b", "b")
        w1.expose_output("s1", "out", "out")
        w1.save(tmp_path / "wf_ckpt")

        w2 = Workflow()
        w2.add_node("s1", _make_add_hg("hg_add", 0, registry))
        w2.expose_input("s1", "a", "a")
        w2.expose_input("s1", "b", "b")
        w2.expose_output("s1", "out", "out")
        w2.load_from_checkpoint(tmp_path / "wf_ckpt")

        result = w2.run({"a": 0, "b": 0}, validate_before=False)
        assert result["out"] == 77


class TestWorkflowCycle:
    def test_cycle_between_hypergraphs(self, registry):
        clear_plan_cache()
        w = Workflow()
        w.add_node("a", _make_identity_hg("cyc_a", registry))
        w.add_node("b", _make_identity_hg("cyc_b", registry))
        w.add_edge("a", "out", "b", "in")
        w.add_edge("b", "out", "a", "in")
        w.expose_input("a", "in", "x")
        w.expose_output("b", "out", "y")
        w.metadata = {"num_loop_steps": 3}

        result = w.run({"x": "loop"}, validate_before=False, num_loop_steps=3)
        assert result["y"] == "loop"
