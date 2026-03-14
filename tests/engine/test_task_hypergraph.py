import pytest
from yggdrasill.engine.planner import clear_plan_cache
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.foundation.registry import BlockRegistry
from tests.foundation.helpers import IdentityTaskNode, AddTaskNode


@pytest.fixture()
def registry():
    r = BlockRegistry()
    r.register("test/identity_task", IdentityTaskNode)
    r.register("test/add_task", AddTaskNode)
    return r


CHAIN_CONFIG = {
    "graph_id": "test_chain",
    "nodes": [
        {"node_id": "A", "block_type": "test/identity_task"},
        {"node_id": "B", "block_type": "test/identity_task"},
    ],
    "edges": [
        {"source_node": "A", "source_port": "out", "target_node": "B", "target_port": "in"},
    ],
    "exposed_inputs": [{"node_id": "A", "port_name": "in", "name": "x"}],
    "exposed_outputs": [{"node_id": "B", "port_name": "out", "name": "y"}],
}


class TestAddNodeFromConfig:
    def test_basic(self, registry):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node_from_config("A", "test/identity_task", registry=registry)
        node = h.get_node("A")
        assert node is not None
        assert node.node_id == "A"
        assert node.block_type == "test/identity_task"

    def test_with_pretrained(self, registry):
        h = Hypergraph()
        h.add_node_from_config("A", "test/add_task", config={"offset": 0},
                               pretrained={"offset": 99}, registry=registry)
        node = h.get_node("A")
        assert node.offset == 99

    def test_duplicate_node_id(self, registry):
        h = Hypergraph()
        h.add_node_from_config("A", "test/identity_task", registry=registry)
        with pytest.raises(ValueError, match="already exists"):
            h.add_node_from_config("A", "test/identity_task", registry=registry)

    def test_unknown_block_type(self, registry):
        h = Hypergraph()
        with pytest.raises(KeyError):
            h.add_node_from_config("A", "nonexistent", registry=registry)


class TestFromConfig:
    def test_basic(self, registry):
        clear_plan_cache()
        h = Hypergraph.from_config(CHAIN_CONFIG, registry=registry)
        assert h.graph_id == "test_chain"
        assert "A" in h.node_ids
        assert "B" in h.node_ids
        assert len(h.get_edges()) == 1
        assert len(h.get_input_spec()) == 1
        assert len(h.get_output_spec()) == 1

    def test_run_after_from_config(self, registry):
        clear_plan_cache()
        h = Hypergraph.from_config(CHAIN_CONFIG, registry=registry)
        out = h.run({"x": 42})
        assert out["y"] == 42

    def test_validate_on_from_config(self, registry):
        bad_config = {
            "nodes": [{"node_id": "A", "block_type": "test/identity_task"}],
            "edges": [],
            "exposed_inputs": [],
            "exposed_outputs": [],
        }
        with pytest.raises(ValueError, match="validation failed"):
            Hypergraph.from_config(bad_config, registry=registry, validate=True)


class TestToConfig:
    def test_roundtrip(self, registry):
        clear_plan_cache()
        h1 = Hypergraph.from_config(CHAIN_CONFIG, registry=registry)
        cfg = h1.to_config()
        assert cfg["schema_version"] == "1.0"
        assert cfg["graph_id"] == "test_chain"
        assert len(cfg["nodes"]) == 2
        assert len(cfg["edges"]) == 1

        h2 = Hypergraph.from_config(cfg, registry=registry)
        assert h2.node_ids == h1.node_ids
        assert h2.run({"x": 7}) == h1.run({"x": 7})


class TestInferExposedPorts:
    def test_infer(self, registry):
        from yggdrasill.engine.edge import Edge
        h = Hypergraph()
        h.add_node_from_config("A", "test/identity_task", registry=registry)
        h.add_node_from_config("B", "test/identity_task", registry=registry)
        h.add_node_from_config("C", "test/identity_task", registry=registry)
        h.add_edge(Edge("A", "out", "B", "in"))
        h.add_edge(Edge("B", "out", "C", "in"))
        h.infer_exposed_ports()
        in_spec = h.get_input_spec()
        out_spec = h.get_output_spec()
        in_ports = {(e["node_id"], e["port_name"]) for e in in_spec}
        out_ports = {(e["node_id"], e["port_name"]) for e in out_spec}
        assert ("A", "in") in in_ports
        assert ("C", "out") in out_ports


class TestStateDictHypergraph:
    def test_state_dict(self, registry):
        h = Hypergraph()
        h.add_node_from_config("A", "test/add_task", config={"offset": 5}, registry=registry)
        sd = h.state_dict()
        assert "A" in sd
        assert sd["A"]["offset"] == 5

    def test_load_state_dict(self, registry):
        clear_plan_cache()
        h = Hypergraph.from_config({
            "nodes": [
                {"node_id": "A", "block_type": "test/add_task", "config": {"offset": 0}},
            ],
            "edges": [],
            "exposed_inputs": [
                {"node_id": "A", "port_name": "a", "name": "a"},
                {"node_id": "A", "port_name": "b", "name": "b"},
            ],
            "exposed_outputs": [{"node_id": "A", "port_name": "out", "name": "out"}],
        }, registry=registry)
        h.load_state_dict({"A": {"offset": 100}})
        out = h.run({"a": 0, "b": 0}, validate_before=False)
        assert out["out"] == 100

    def test_get_input_spec_with_dtype(self, registry):
        h = Hypergraph.from_config(CHAIN_CONFIG, registry=registry)
        spec = h.get_input_spec(include_dtype=True)
        assert "dtype" in spec[0]
