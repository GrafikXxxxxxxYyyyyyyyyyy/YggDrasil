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

class TestMissingCheckpoint:
    """Spec PHASE_5 §9.2: loading with missing checkpoint returns graph without weights."""

    def test_load_missing_checkpoint_no_error(self, tmp_path, registry, chain_config):
        h = Hypergraph.from_config(chain_config, registry=registry)
        save_hypergraph(h, tmp_path / "hg")
        (tmp_path / "hg" / "checkpoint.pkl").unlink()
        h2 = load_hypergraph(tmp_path / "hg", registry=registry)
        assert h2.node_ids == h.node_ids

    def test_load_flag_false_skips_checkpoint(self, tmp_path, registry, chain_config):
        h = Hypergraph.from_config(chain_config, registry=registry)
        save_hypergraph(h, tmp_path / "hg")
        h2 = load_hypergraph(
            tmp_path / "hg", registry=registry, load_checkpoint_flag=False,
        )
        assert h2.node_ids == h.node_ids


class TestHypergraphSaveLoadMethods:
    """Test save/load/save_config/load_config/load_from_checkpoint ON Hypergraph."""

    def test_save_load_roundtrip(self, tmp_path, registry, chain_config):
        clear_plan_cache()
        h1 = Hypergraph.from_config(chain_config, registry=registry)
        out1 = h1.run({"x": 1, "b": 2}, validate_before=False)
        h1.save(tmp_path / "hg_m")
        h2 = Hypergraph.load(tmp_path / "hg_m", registry=registry)
        out2 = h2.run({"x": 1, "b": 2}, validate_before=False)
        assert out1 == out2

    def test_save_config_load_config(self, tmp_path, registry, chain_config):
        h1 = Hypergraph.from_config(chain_config, registry=registry)
        h1.save_config(tmp_path / "hg_c")
        h2 = Hypergraph.load_config(tmp_path / "hg_c", registry=registry)
        assert h2.node_ids == h1.node_ids

    def test_load_from_checkpoint(self, tmp_path, registry, chain_config):
        clear_plan_cache()
        h1 = Hypergraph.from_config(chain_config, registry=registry)
        h1.save(tmp_path / "hg_fc")
        h2 = Hypergraph.load_config(tmp_path / "hg_fc", registry=registry)
        h2.load_from_checkpoint(tmp_path / "hg_fc")
        out1 = h1.run({"x": 1, "b": 2}, validate_before=False)
        out2 = h2.run({"x": 1, "b": 2}, validate_before=False)
        assert out1 == out2


class TestYAMLImportError:
    def test_read_config_yaml_without_pyyaml(self, tmp_path, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("no yaml")
            return real_import(name, *args, **kwargs)

        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("{}")
        monkeypatch.setattr(builtins, "__import__", mock_import)
        from yggdrasill.hypergraph.serialization import _read_config
        with pytest.raises(ImportError, match="PyYAML"):
            _read_config(yaml_path)


class TestYAMLConfigIO:
    def test_read_yaml_config(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        data = {"foo": 1, "bar": [1, 2]}
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(data))
        from yggdrasill.hypergraph.serialization import _read_config
        loaded = _read_config(yaml_path)
        assert loaded == data

    def test_read_yml_extension(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        data = {"key": "value"}
        yml_path = tmp_path / "config.yml"
        yml_path.write_text(yaml.dump(data))
        from yggdrasill.hypergraph.serialization import _read_config
        loaded = _read_config(yml_path)
        assert loaded == data


class TestSerializationDeduplicateState:
    def test_deduplicate_shared_block_ids(self, tmp_path, registry):
        from tests.foundation.helpers import AddTaskNode
        from yggdrasill.hypergraph.serialization import _deduplicate_state, _expand_deduped_state
        h = Hypergraph()
        n1 = AddTaskNode(node_id="X", block_id="shared", config={"offset": 10})
        n2 = AddTaskNode(node_id="Y", block_id="shared", config={"offset": 10})
        h.add_node("X", n1)
        h.add_node("Y", n2)
        raw_state = {"X": n1.state_dict(), "Y": n2.state_dict()}
        deduped = _deduplicate_state(h, raw_state)
        assert "_aliases" in deduped
        assert "Y" not in {k for k in deduped if k != "_aliases"}
        expanded = _expand_deduped_state(h, deduped)
        assert "X" in expanded
        assert "Y" in expanded
        assert expanded["X"] == expanded["Y"]

    def test_load_checkpoint_public(self, tmp_path, registry):
        from yggdrasill.hypergraph.serialization import save_checkpoint, load_checkpoint
        save_checkpoint({"A": {"offset": 7}}, tmp_path / "ckpt.pkl")
        loaded = load_checkpoint(tmp_path / "ckpt.pkl")
        assert loaded == {"A": {"offset": 7}}


class TestSerializationSchemaVersionWarning:
    def test_load_block_warns_on_mismatched_schema(self, tmp_path, registry):
        import warnings
        node = registry.build({"block_type": "test/identity_task", "node_id": "N"})
        save_block(node, tmp_path / "block")
        cfg = load_config(tmp_path / "block" / "config.json")
        cfg["schema_version"] = "9.9"
        from yggdrasill.hypergraph.serialization import save_config as _sc
        _sc(cfg, tmp_path / "block" / "config.json")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_block(tmp_path / "block", registry=registry)
            schema_warnings = [x for x in w if "schema_version" in str(x.message)]
            assert len(schema_warnings) >= 1

    def test_load_hypergraph_warns_on_mismatched_schema(self, tmp_path, registry, chain_config):
        import warnings
        h = Hypergraph.from_config(chain_config, registry=registry)
        save_hypergraph(h, tmp_path / "hg")
        cfg = load_config(tmp_path / "hg" / "config.json")
        cfg["schema_version"] = "9.9"
        from yggdrasill.hypergraph.serialization import save_config as _sc
        _sc(cfg, tmp_path / "hg" / "config.json")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_hypergraph(tmp_path / "hg", registry=registry)
            schema_warnings = [x for x in w if "schema_version" in str(x.message)]
            assert len(schema_warnings) >= 1


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
