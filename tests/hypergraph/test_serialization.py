import pytest
from yggdrasill.engine.planner import clear_plan_cache
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.foundation.registry import BlockRegistry
from yggdrasill.hypergraph.serialization import (
    load_block,
    load_config,
    load_hypergraph,
    save_block,
    save_config,
    save_hypergraph,
)
from tests.foundation.helpers import AddTaskNode, IdentityTaskNode

import yggdrasill.task_nodes.stubs  # noqa: F401 -- ensure stubs registered


@pytest.fixture()
def registry():
    r = BlockRegistry()
    r.register("test/identity_task", IdentityTaskNode)
    r.register("test/add_task", AddTaskNode)
    return r


@pytest.fixture()
def chain_config():
    return {
        "graph_id": "ser_chain",
        "nodes": [
            {"node_id": "A", "block_type": "test/identity_task"},
            {"node_id": "B", "block_type": "test/add_task", "config": {"offset": 10}},
        ],
        "edges": [
            {"source_node": "A", "source_port": "out", "target_node": "B", "target_port": "a"},
        ],
        "exposed_inputs": [
            {"node_id": "A", "port_name": "in", "name": "x"},
            {"node_id": "B", "port_name": "b", "name": "b"},
        ],
        "exposed_outputs": [{"node_id": "B", "port_name": "out", "name": "out"}],
    }


# ---------------------------------------------------------------------------
# Low-level config / checkpoint
# ---------------------------------------------------------------------------

class TestConfigIO:
    def test_save_load_config(self, tmp_path):
        data = {"foo": 1, "bar": [1, 2, 3]}
        save_config(data, tmp_path / "test.json")
        loaded = load_config(tmp_path / "test.json")
        assert loaded == data


# ---------------------------------------------------------------------------
# Block-level save / load
# ---------------------------------------------------------------------------

class TestBlockSerialization:
    def test_save_load_identity(self, tmp_path, registry):
        node = registry.build({"block_type": "test/identity_task", "node_id": "N1", "block_id": "b1"})
        save_block(node, tmp_path / "block")
        loaded = load_block(tmp_path / "block", registry=registry)
        assert loaded.block_type == "test/identity_task"

    def test_save_load_add_with_state(self, tmp_path, registry):
        node = registry.build({
            "block_type": "test/add_task",
            "node_id": "N2",
            "block_id": "a2",
            "offset": 77,
        })
        save_block(node, tmp_path / "add_block")
        loaded = load_block(tmp_path / "add_block", registry=registry)
        assert loaded.forward({"a": 0, "b": 0}) == {"out": 77}

    def test_config_has_schema_version(self, tmp_path, registry):
        node = registry.build({"block_type": "test/identity_task", "node_id": "N"})
        save_block(node, tmp_path / "block")
        cfg = load_config(tmp_path / "block" / "config.json")
        assert "schema_version" in cfg


# ---------------------------------------------------------------------------
# Hypergraph-level save / load
# ---------------------------------------------------------------------------

class TestHypergraphSerialization:
    def test_roundtrip(self, tmp_path, registry, chain_config):
        clear_plan_cache()
        h1 = Hypergraph.from_config(chain_config, registry=registry)
        out1 = h1.run({"x": 1, "b": 2}, validate_before=False)

        save_hypergraph(h1, tmp_path / "hg")
        h2 = load_hypergraph(tmp_path / "hg", registry=registry)
        out2 = h2.run({"x": 1, "b": 2}, validate_before=False)
        assert out1 == out2

    def test_config_file_exists(self, tmp_path, registry, chain_config):
        h = Hypergraph.from_config(chain_config, registry=registry)
        save_hypergraph(h, tmp_path / "hg")
        assert (tmp_path / "hg" / "config.json").exists()

    def test_checkpoint_created_when_state(self, tmp_path, registry, chain_config):
        h = Hypergraph.from_config(chain_config, registry=registry)
        save_hypergraph(h, tmp_path / "hg")
        assert (tmp_path / "hg" / "checkpoint.pkl").exists()

    def test_schema_version(self, tmp_path, registry, chain_config):
        h = Hypergraph.from_config(chain_config, registry=registry)
        save_hypergraph(h, tmp_path / "hg")
        cfg = load_config(tmp_path / "hg" / "config.json")
        assert cfg["schema_version"] == "1.0"


# ---------------------------------------------------------------------------
# block_id deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_shared_block_id(self, tmp_path, registry):
        clear_plan_cache()
        cfg = {
            "graph_id": "dedup",
            "nodes": [
                {"node_id": "N1", "block_type": "test/add_task", "block_id": "shared", "config": {"offset": 42}},
                {"node_id": "N2", "block_type": "test/add_task", "block_id": "shared", "config": {"offset": 42}},
            ],
            "edges": [],
            "exposed_inputs": [
                {"node_id": "N1", "port_name": "a", "name": "a1"},
                {"node_id": "N1", "port_name": "b", "name": "b1"},
                {"node_id": "N2", "port_name": "a", "name": "a2"},
                {"node_id": "N2", "port_name": "b", "name": "b2"},
            ],
            "exposed_outputs": [
                {"node_id": "N1", "port_name": "out", "name": "out1"},
                {"node_id": "N2", "port_name": "out", "name": "out2"},
            ],
        }
        h = Hypergraph.from_config(cfg, registry=registry)
        save_hypergraph(h, tmp_path / "dedup")
        h2 = load_hypergraph(tmp_path / "dedup", registry=registry)

        r1 = h2.run({"a1": 0, "b1": 0, "a2": 0, "b2": 0}, validate_before=False)
        assert r1["out1"] == 42
        assert r1["out2"] == 42


# ---------------------------------------------------------------------------
# Roundtrip: save -> load -> run == original run
# ---------------------------------------------------------------------------

class TestRoundtripFullPipeline:
    def test_full_roundtrip(self, tmp_path):
        clear_plan_cache()
        reg = BlockRegistry.global_registry()
        cfg = {
            "graph_id": "pipe",
            "nodes": [
                {"node_id": "enc", "block_type": "converter/identity"},
                {"node_id": "bb", "block_type": "backbone/identity"},
                {"node_id": "dec", "block_type": "converter/identity"},
            ],
            "edges": [
                {"source_node": "enc", "source_port": "output", "target_node": "bb", "target_port": "latent"},
                {"source_node": "bb", "source_port": "pred", "target_node": "dec", "target_port": "input"},
            ],
            "exposed_inputs": [
                {"node_id": "enc", "port_name": "input", "name": "x"},
                {"node_id": "bb", "port_name": "timestep", "name": "timestep"},
            ],
            "exposed_outputs": [{"node_id": "dec", "port_name": "output", "name": "y"}],
        }
        h1 = Hypergraph.from_config(cfg, registry=reg)
        out1 = h1.run({"x": "data", "timestep": 0})

        save_hypergraph(h1, tmp_path / "pipe")
        h2 = load_hypergraph(tmp_path / "pipe", registry=reg)
        out2 = h2.run({"x": "data", "timestep": 0})
        assert out1 == out2
