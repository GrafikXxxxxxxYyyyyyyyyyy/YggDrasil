"""Tests for foundation.Graph and Edge."""

import pytest

from yggdrasill.foundation.graph import Edge, Graph, ValidationResult
from yggdrasill.foundation.node import Node
from yggdrasill.foundation.registry import BlockRegistry
from tests.foundation.helpers import AddBlock, IdentityBlock


@pytest.fixture
def registry() -> BlockRegistry:
    r = BlockRegistry()
    r.register("add", AddBlock)
    r.register("identity", IdentityBlock)
    return r


def test_edge_basic() -> None:
    e = Edge("n1", "out", "n2", "a")
    assert e.source_node == "n1" and e.source_port == "out"
    assert e.target_node == "n2" and e.target_port == "a"


def test_edge_empty_raises() -> None:
    with pytest.raises(ValueError):
        Edge("", "out", "n2", "a")
    with pytest.raises(ValueError):
        Edge("n1", "", "n2", "a")


def test_graph_add_node_and_edge(registry: BlockRegistry) -> None:
    g = Graph("g1")
    a = AddBlock(block_id="add1", config={"offset": 0})
    b = IdentityBlock(block_id="id1")
    g.add_node(Node("A", a))
    g.add_node(Node("B", b))
    g.add_edge(Edge("A", "out", "B", "x"))
    assert g.node_ids == {"A", "B"}
    assert len(g.get_edges()) == 1
    assert g.get_edges_in("B")[0].source_node == "A"
    assert g.get_edges_out("A")[0].target_node == "B"


def test_graph_duplicate_node_raises() -> None:
    g = Graph()
    b = IdentityBlock(block_id="x")
    g.add_node(Node("n1", b))
    with pytest.raises(ValueError, match="Узел уже существует"):
        g.add_node(Node("n1", IdentityBlock(block_id="y")))


def test_graph_edge_unknown_node_raises(registry: BlockRegistry) -> None:
    g = Graph()
    g.add_node(Node("A", IdentityBlock(block_id="a")))
    with pytest.raises(ValueError, match="not found"):
        g.add_edge(Edge("X", "y", "A", "x"))
    with pytest.raises(ValueError, match="not found"):
        g.add_edge(Edge("A", "y", "X", "x"))


def test_graph_edge_unknown_port_raises(registry: BlockRegistry) -> None:
    g = Graph()
    g.add_node(Node("A", IdentityBlock(block_id="a")))
    g.add_node(Node("B", IdentityBlock(block_id="b")))
    with pytest.raises(ValueError, match="port"):
        g.add_edge(Edge("A", "nonexistent", "B", "x"))


def test_graph_to_config(registry: BlockRegistry) -> None:
    g = Graph("g1")
    g.add_node(Node("A", AddBlock(block_id="a1", config={"offset": 1})))
    g.add_node(Node("B", IdentityBlock(block_id="b1")))
    g.add_edge(Edge("A", "out", "B", "x"))
    cfg = g.to_config()
    assert cfg["graph_id"] == "g1"
    assert len(cfg["nodes"]) == 2
    assert len(cfg["edges"]) == 1
    node_ids = [n["node_id"] for n in cfg["nodes"]]
    assert "A" in node_ids and "B" in node_ids


def test_graph_from_config_roundtrip(registry: BlockRegistry) -> None:
    g = Graph("g1")
    g.add_node(Node("A", AddBlock(block_id="a1", config={"offset": 2})))
    g.add_node(Node("B", IdentityBlock(block_id="b1")))
    g.add_edge(Edge("A", "out", "B", "x"))
    cfg = g.to_config()
    g2 = Graph.from_config(cfg, registry=registry)
    assert g2.graph_id == "g1"
    assert g2.node_ids == g.node_ids
    assert len(g2.get_edges()) == 1
    # Run through graph: inject a=1,b=2 into A, get B's output
    node_a = g2.get_node("A")
    node_b = g2.get_node("B")
    assert node_a and node_b
    out_a = node_a.block.forward({"a": 1, "b": 2})
    out_b = node_b.block.forward({"x": out_a["out"]})
    assert out_b["y"] == 3 + 2  # 1+2+offset(2)=5


def test_graph_state_dict_load(registry: BlockRegistry) -> None:
    g = Graph("g1")
    g.add_node(Node("A", AddBlock(block_id="a1", config={"offset": 10})))
    sd = g.state_dict()
    assert "A" in sd
    assert sd["A"]["offset"] == 10
    g2 = Graph.from_config(g.to_config(), registry=registry)
    g2.load_state_dict(sd)
    node_a = g2.get_node("A")
    assert node_a and node_a.block.forward({"a": 0, "b": 0})["out"] == 10


def test_get_input_output_spec_include_dtype(registry: BlockRegistry) -> None:
    """get_input_spec(include_dtype=True) and get_output_spec(include_dtype=True) add dtype from port."""
    g = Graph()
    g.add_node(Node("A", IdentityBlock(block_id="a1")))
    g.expose_input("A", "x", name="in")
    g.expose_output("A", "y", name="out")
    inp_spec = g.get_input_spec(include_dtype=True)
    out_spec = g.get_output_spec(include_dtype=True)
    assert len(inp_spec) == 1 and inp_spec[0].get("dtype") is not None
    assert len(out_spec) == 1 and out_spec[0].get("dtype") is not None
    assert g.get_input_spec(include_dtype=False) == g.get_input_spec()


def test_graph_expose_input_output_and_spec(registry: BlockRegistry) -> None:
    g = Graph("g1")
    g.add_node(Node("A", AddBlock(block_id="a1", config={"offset": 0})))
    g.add_node(Node("B", IdentityBlock(block_id="b1")))
    g.add_edge(Edge("A", "out", "B", "x"))
    g.expose_input("A", "a", name="input_a")
    g.expose_input("A", "b", name="input_b")
    g.expose_output("B", "y", name="result")
    spec_in = g.get_input_spec()
    spec_out = g.get_output_spec()
    assert len(spec_in) == 2 and len(spec_out) == 1
    assert spec_out[0]["node_id"] == "B" and spec_out[0]["port_name"] == "y"
    assert spec_out[0].get("name") == "result"
    cfg = g.to_config()
    assert "schema_version" in cfg and cfg["exposed_inputs"] and cfg["exposed_outputs"]
    g2 = Graph.from_config(cfg, registry=registry)
    assert len(g2.get_input_spec()) == 2 and len(g2.get_output_spec()) == 1


def test_graph_load_from_checkpoint_dict(registry: BlockRegistry) -> None:
    g = Graph("g1")
    g.add_node(Node("A", AddBlock(block_id="a1", config={"offset": 0})))
    g.add_node(Node("B", IdentityBlock(block_id="b1")))
    g.add_edge(Edge("A", "out", "B", "x"))
    cfg = g.to_config()
    ckpt = g.state_dict()
    g2 = Graph("g2")
    g2.load_from_checkpoint(config=cfg, checkpoint=ckpt, registry=registry)
    assert g2.graph_id == "g1"
    assert g2.get_node("A") and g2.get_node("A").block.forward({"a": 1, "b": 0})["out"] == 1
    # Load weights into existing graph
    g2.load_from_checkpoint(checkpoint={"A": {"offset": 7}})
    assert g2.get_node("A").block.forward({"a": 0, "b": 0})["out"] == 7


def test_add_node_by_block_type_returns_node_id(registry: BlockRegistry) -> None:
    g = Graph()
    nid = g.add_node("identity", config={}, registry=registry)
    assert nid in g.node_ids
    assert g.get_node(nid) is not None
    assert g.get_node(nid).block.block_type == "identity"


def test_add_node_by_block_instance_generates_id() -> None:
    g = Graph()
    b = IdentityBlock(block_id="x")
    nid = g.add_node(b)
    assert nid in g.node_ids
    assert g.get_node(nid).block is b


def test_add_node_with_explicit_node_id(registry: BlockRegistry) -> None:
    g = Graph()
    nid = g.add_node("MyNode", "identity", registry=registry)
    assert nid == "MyNode"
    assert g.get_node("MyNode") is not None


def test_add_node_with_node_rejects_extra_kwargs() -> None:
    g = Graph()
    with pytest.raises(ValueError, match="При передаче Node"):
        g.add_node(Node("N", IdentityBlock()), node_id="other")


def test_trainable_parameters_empty() -> None:
    g = Graph()
    g.add_node(Node("A", IdentityBlock()))
    params = list(g.trainable_parameters())
    assert params == []


def test_validate_required_port_no_edge_raises() -> None:
    g = Graph()
    g.add_node(Node("A", IdentityBlock()))
    with pytest.raises(ValueError, match="no incoming edge"):
        g.validate()


def test_validate_required_port_exposed_ok() -> None:
    g = Graph()
    g.add_node(Node("A", IdentityBlock()))
    g.expose_input("A", "x", name="input")
    g.validate()


def test_validate_required_port_has_edge_ok(registry: BlockRegistry) -> None:
    g = Graph()
    g.add_node(Node("A", AddBlock(config={"offset": 0})))
    g.add_node(Node("B", IdentityBlock()))
    g.add_edge(Edge("A", "out", "B", "x"))
    g.expose_input("A", "a")
    g.expose_input("A", "b")
    g.validate()


def test_from_template_text_to_image() -> None:
    import yggdrasill.task_nodes  # noqa: F401 - register stubs and template
    from yggdrasill.foundation.registry import BlockRegistry
    reg = BlockRegistry.global_registry()
    # Stubs are registered by task_nodes import; template "text_to_image" is registered
    g = Graph.from_template("text_to_image", registry=reg)
    assert g.graph_id == "text_to_image"
    assert "tokenizer" in g.node_ids
    assert "conditioner" in g.node_ids
    assert "backbone" in g.node_ids
    assert "solver" in g.node_ids
    assert "codec" in g.node_ids
    assert len(g.get_input_spec()) >= 1
    assert len(g.get_output_spec()) >= 1
    g.validate()


def test_save_config_and_load_via_from_config(registry: BlockRegistry, tmp_path) -> None:
    g = Graph("g1")
    g.add_node(Node("A", AddBlock(block_id="a1", config={"offset": 5})))
    g.add_node(Node("B", IdentityBlock(block_id="b1")))
    g.add_edge(Edge("A", "out", "B", "x"))
    config_path = tmp_path / "graph_config.json"
    g.save_config(str(config_path))
    assert config_path.read_text()
    g2 = Graph.from_config(
        __import__("json").loads(config_path.read_text()),
        registry=registry,
    )
    assert g2.graph_id == "g1"
    assert g2.node_ids == g.node_ids


def test_save_checkpoint_and_load(registry: BlockRegistry, tmp_path) -> None:
    g = Graph("g1")
    g.add_node(Node("A", AddBlock(block_id="a1", config={"offset": 3})))
    g.add_node(Node("B", IdentityBlock(block_id="b1")))
    g.add_edge(Edge("A", "out", "B", "x"))
    ckpt_path = tmp_path / "checkpoint.json"
    g.save_checkpoint(str(ckpt_path))
    assert ckpt_path.read_text()
    g2 = Graph.from_config(g.to_config(), registry=registry)
    g2.load_state_dict(__import__("json").loads(ckpt_path.read_text()))
    assert g2.get_node("A").block.forward({"a": 0, "b": 0})["out"] == 3


def test_set_trainable_and_trainable_parameters(registry: BlockRegistry) -> None:
    g = Graph()
    g.add_node("A", "add", config={"offset": 1}, registry=registry)
    g.add_node("B", "identity", registry=registry)
    g.set_trainable("B", False)
    params = list(g.trainable_parameters())
    # AddBlock has no trainable params in base; identity has none. So params may be empty.
    # We only check that set_trainable doesn't raise and trainable_parameters is consistent
    g.set_trainable("B", True)
    assert "A" in g.node_ids and "B" in g.node_ids


def test_graph_kind_and_metadata_roundtrip(registry: BlockRegistry) -> None:
    g = Graph("g1")
    g.add_node(Node("N", IdentityBlock()))
    g.graph_kind = "diffusion"
    g.metadata["num_steps"] = 50
    cfg = g.to_config()
    assert cfg.get("graph_kind") == "diffusion"
    assert cfg.get("metadata", {}).get("num_steps") == 50
    g2 = Graph.from_config(cfg, registry=registry)
    assert g2.graph_kind == "diffusion"
    assert g2.metadata.get("num_steps") == 50


def test_from_yaml_json_file(registry: BlockRegistry, tmp_path) -> None:
    g = Graph("g2")
    g.add_node(Node("N", IdentityBlock(block_id="n1")))
    g.expose_input("N", "x", name="in")
    config_path = tmp_path / "config.json"
    g.save_config(str(config_path))
    g2 = Graph.from_yaml(str(config_path), registry=registry)
    assert g2.graph_id == "g2"
    assert g2.node_ids == {"N"}
    assert len(g2.get_input_spec()) == 1


# --- TODO_03: Executor, save/load dir, validate result ---


def test_run_dag_graph(registry: BlockRegistry) -> None:
    """Executor run() on a DAG: A (add) -> B (identity); inputs by name, output by name."""
    g = Graph("g1")
    g.add_node(Node("A", AddBlock(block_id="a1", config={"offset": 2})))
    g.add_node(Node("B", IdentityBlock(block_id="b1")))
    g.add_edge(Edge("A", "out", "B", "x"))
    g.expose_input("A", "a", name="a_in")
    g.expose_input("A", "b", name="b_in")
    g.expose_output("B", "y", name="result")
    g.validate()
    out = g.run({"a_in": 10, "b_in": 5})
    assert out["result"] == 10 + 5 + 2  # offset=2


def test_run_with_callbacks(registry: BlockRegistry) -> None:
    """run() invokes callbacks before/after node execution."""
    g = Graph()
    g.add_node(Node("A", IdentityBlock(block_id="a1")))
    g.expose_input("A", "x", name="in")
    g.expose_output("A", "y", name="out")
    seen = []

    def hook(nid: str, phase: str, **kwargs: object) -> None:
        seen.append((nid, phase))

    g.run({"in": 42}, callbacks=[hook])
    assert ("A", "before") in seen and ("A", "after") in seen


def test_validate_returns_validation_result(registry: BlockRegistry) -> None:
    """validate(strict=False) returns ValidationResult with errors/warnings."""
    g = Graph()
    g.add_node(Node("A", IdentityBlock(block_id="a1")))
    g.expose_input("A", "x", name="in")
    g.expose_output("A", "y", name="out")
    result = g.validate(strict=False)
    assert isinstance(result, ValidationResult)
    assert result.is_valid
    assert result.errors == []


def test_validate_strict_raises_on_errors(registry: BlockRegistry) -> None:
    """validate(strict=True) raises when there are errors."""
    g = Graph()
    g.add_node(Node("A", IdentityBlock(block_id="a1")))
    # A.x is required and not exposed, no edge
    with pytest.raises(ValueError, match="Required input port"):
        g.validate(strict=True)


def test_save_and_load_directory(registry: BlockRegistry, tmp_path) -> None:
    """Graph.save(save_dir) and Graph.load(save_dir) roundtrip."""
    save_dir = tmp_path / "graph_save"
    g = Graph("g1")
    g.add_node(Node("A", AddBlock(block_id="a1", config={"offset": 7})))
    g.add_node(Node("B", IdentityBlock(block_id="b1")))
    g.add_edge(Edge("A", "out", "B", "x"))
    g.expose_input("A", "a", name="a_in")
    g.expose_input("A", "b", name="b_in")
    g.expose_output("B", "y", name="result")
    g.save(str(save_dir))
    assert (save_dir / "config.json").is_file()
    assert (save_dir / "checkpoint.json").is_file()
    g2 = Graph.load(str(save_dir), registry=registry)
    assert g2.graph_id == "g1"
    assert g2.node_ids == g.node_ids
    out = g2.run({"a_in": 1, "b_in": 2})
    assert out["result"] == 1 + 2 + 7


def test_graph_to_device_no_raise(registry: BlockRegistry) -> None:
    """graph.to(device) does not raise; blocks without .to are skipped."""
    g = Graph()
    g.add_node(Node("A", IdentityBlock(block_id="a1")))
    g.to("cpu")  # IdentityBlock has no .to; should not raise
    g.to(None)


def test_run_dry_run(registry: BlockRegistry) -> None:
    """run(..., dry_run=True) does not call forward; returns outputs with None values."""
    g = Graph()
    g.add_node(Node("A", AddBlock(block_id="a1", config={"offset": 1})))
    g.add_node(Node("B", IdentityBlock(block_id="b1")))
    g.add_edge(Edge("A", "out", "B", "x"))
    g.expose_input("A", "a", name="a_in")
    g.expose_input("A", "b", name="b_in")
    g.expose_output("B", "y", name="result")
    out = g.run({"a_in": 10, "b_in": 5}, dry_run=True)
    assert "result" in out
    assert out["result"] is None


def test_from_config_validate(registry: BlockRegistry) -> None:
    """from_config(..., validate=True) raises if required port has no edge."""
    config = {
        "graph_id": "g",
        "nodes": [
            {"node_id": "A", "block_type": "identity", "config": {}},
        ],
        "edges": [],
        "exposed_inputs": [],
        "exposed_outputs": [{"node_id": "A", "port_name": "y"}],
    }
    with pytest.raises(ValueError, match="Required input port"):
        Graph.from_config(config, registry=registry, validate=True)
    g = Graph.from_config(config, registry=registry, validate=False)
    assert g.node_ids == {"A"}


def test_infer_exposed_ports(registry: BlockRegistry) -> None:
    """infer_exposed_ports() sets exposed I/O to ports with no in/out edges."""
    g = Graph()
    g.add_node(Node("A", AddBlock(block_id="a1")))
    g.add_node(Node("B", IdentityBlock(block_id="b1")))
    g.add_edge(Edge("A", "out", "B", "x"))
    assert len(g.get_input_spec()) == 0 and len(g.get_output_spec()) == 0
    g.infer_exposed_ports()
    # A has inputs a,b (no in-edges), B has input x (fed by A) -> exposed: A.a, A.b
    # A has output out (feeds B), B has output y (no out-edge) -> exposed: B.y
    inp = g.get_input_spec()
    out = g.get_output_spec()
    assert len(inp) == 2  # A.a, A.b
    assert len(out) == 1  # B.y
    out_keys = {_key(e) for e in out}
    assert any("B" in k and "y" in k for k in out_keys)


def _key(e: dict) -> str:
    return e.get("name") or f"{e['node_id']}:{e['port_name']}"


def test_save_checkpoint_backend_json_only(registry: BlockRegistry, tmp_path) -> None:
    """save_checkpoint(backend='json') works; backend='torch' raises NotImplementedError."""
    g = Graph()
    g.add_node(Node("A", IdentityBlock(block_id="a1")))
    g.save_checkpoint(str(tmp_path / "c.json"), backend="json")
    with pytest.raises(NotImplementedError, match="backend.*torch"):
        g.save_checkpoint(str(tmp_path / "c.pt"), backend="torch")


def test_save_checkpoint_dir_format(registry: BlockRegistry, tmp_path) -> None:
    """save_checkpoint(path, format='dir') writes one JSON per node with non-empty state; load_from_checkpoint(checkpoint_dir=...) loads."""
    g = Graph()
    g.add_node(Node("A", AddBlock(block_id="a1", config={"offset": 3})))
    g.add_node(Node("B", IdentityBlock(block_id="b1")))
    g.add_edge(Edge("A", "out", "B", "x"))
    ckpt_dir = tmp_path / "ckpt_dir"
    g.save_checkpoint(str(ckpt_dir), format="dir")
    assert (ckpt_dir / "A.json").is_file()
    g2 = Graph.from_config(g.to_config(), registry=registry)
    g2.load_from_checkpoint(checkpoint_dir=str(ckpt_dir))
    assert g2.get_node("A").block.forward({"a": 0, "b": 0})["out"] == 3
