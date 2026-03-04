"""Tests for Pipeline: add_graph, add_edge, validate, run, config roundtrip, save/load."""

import json
import tempfile
from pathlib import Path

import pytest

from yggdrasill.foundation import Graph, Edge
from yggdrasill.foundation.registry import BlockRegistry
from yggdrasill.pipeline import Pipeline, PipelineEdge
from tests.foundation.helpers import AddBlock, IdentityBlock


def _make_single_graph(registry: BlockRegistry, offset: int = 0) -> Graph:
    g = Graph()
    g.add_node("A", "add", config={"offset": offset}, registry=registry)
    g.add_node("B", "identity", registry=registry)
    g.add_edge(Edge("A", "out", "B", "x"))
    g.expose_input("A", "a", name="a_in")
    g.expose_input("A", "b", name="b_in")
    g.expose_output("B", "y", name="result")
    return g


def _make_two_graphs(registry: BlockRegistry):
    g1 = Graph()
    g1.add_node("A", "add", config={"offset": 0}, registry=registry)
    g1.add_node("B", "identity", registry=registry)
    g1.add_edge(Edge("A", "out", "B", "x"))
    g1.expose_input("A", "a", name="a_in")
    g1.expose_input("A", "b", name="b_in")
    g1.expose_output("B", "y", name="result")

    g2 = Graph()
    g2.add_node("X", "add", config={"offset": 10}, registry=registry)
    g2.expose_input("X", "a", name="a_in")
    g2.expose_input("X", "b", name="b_in")
    g2.expose_output("X", "out", name="result")
    return g1, g2


# --- add_graph ---


def test_pipeline_add_graph_instance(registry: BlockRegistry) -> None:
    g = _make_single_graph(registry)
    p = Pipeline()
    nid = p.add_graph(g, "G1")
    assert nid == "G1"
    assert p.node_ids == {"G1"}
    assert p.get_graph("G1") is g


def test_pipeline_add_graph_auto_node_id(registry: BlockRegistry) -> None:
    g = Graph("my_graph")
    g.add_node("A", "add", config={"offset": 0}, registry=registry)
    g.add_node("B", "identity", registry=registry)
    g.add_edge(Edge("A", "out", "B", "x"))
    g.expose_input("A", "a", name="a_in")
    g.expose_input("A", "b", name="b_in")
    g.expose_output("B", "y", name="result")
    p = Pipeline()
    nid = p.add_graph(g)
    assert nid == "my_graph"
    assert p.get_graph("my_graph") is g


def test_pipeline_add_graph_from_config(registry: BlockRegistry) -> None:
    g = _make_single_graph(registry)
    cfg = g.to_config()
    p = Pipeline()
    nid = p.add_graph(cfg, "G1", registry=registry)
    assert nid == "G1"
    assert p.get_graph("G1") is not g
    out = p.get_graph("G1").run({"a_in": 1, "b_in": 2})
    assert out.get("result") == 3


def test_pipeline_add_graph_duplicate_node_id_raises(registry: BlockRegistry) -> None:
    g = _make_single_graph(registry)
    p = Pipeline()
    p.add_graph(g, "G1")
    with pytest.raises(ValueError, match="already exists"):
        p.add_graph(g, "G1")


# --- add_edge ---


def test_pipeline_add_edge_and_validate_ports(registry: BlockRegistry) -> None:
    g1, g2 = _make_two_graphs(registry)
    p = Pipeline()
    p.add_graph(g1, "G1")
    p.add_graph(g2, "G2")
    p.add_edge("G1", "result", "G2", "a_in")
    p.expose_input("G1", "a_in", name="a")
    p.expose_input("G1", "b_in", name="b")
    p.expose_output("G2", "result", name="out")
    edges = p.get_edges()
    assert len(edges) == 1
    assert edges[0].source_node == "G1" and edges[0].source_port == "result"
    assert edges[0].target_node == "G2" and edges[0].target_port == "a_in"


def test_pipeline_add_edge_unknown_source_port_raises(registry: BlockRegistry) -> None:
    g1, g2 = _make_two_graphs(registry)
    p = Pipeline()
    p.add_graph(g1, "G1")
    p.add_graph(g2, "G2")
    with pytest.raises(ValueError, match="Source port"):
        p.add_edge("G1", "nonexistent", "G2", "a_in")


def test_pipeline_add_edge_unknown_target_port_raises(registry: BlockRegistry) -> None:
    g1, g2 = _make_two_graphs(registry)
    p = Pipeline()
    p.add_graph(g1, "G1")
    p.add_graph(g2, "G2")
    with pytest.raises(ValueError, match="Target port"):
        p.add_edge("G1", "result", "G2", "nonexistent")


def test_pipeline_add_edge_unknown_node_raises(registry: BlockRegistry) -> None:
    g1 = _make_single_graph(registry)
    p = Pipeline()
    p.add_graph(g1, "G1")
    with pytest.raises(ValueError, match="not found"):
        p.add_edge("G1", "result", "G2", "a_in")


# --- validate ---


def test_pipeline_validate_dag_ok(registry: BlockRegistry) -> None:
    g1, g2 = _make_two_graphs(registry)
    p = Pipeline()
    p.add_graph(g1, "G1")
    p.add_graph(g2, "G2")
    p.add_edge("G1", "result", "G2", "a_in")
    p.expose_input("G1", "a_in", name="a")
    p.expose_input("G1", "b_in", name="b")
    p.expose_output("G2", "result", name="out")
    result = p.validate(strict=False)
    assert result.is_valid
    assert not result.errors


def test_pipeline_validate_cycle_error(registry: BlockRegistry) -> None:
    g1, g2 = _make_two_graphs(registry)
    p = Pipeline()
    p.add_graph(g1, "G1")
    p.add_graph(g2, "G2")
    p.add_edge("G1", "result", "G2", "a_in")
    p.add_edge("G2", "result", "G1", "a_in")  # cycle
    p.expose_input("G1", "b_in", name="b")
    p.expose_output("G2", "result", name="out")
    result = p.validate(strict=False)
    assert not result.is_valid
    assert any("cycle" in e.lower() for e in result.errors)


def test_pipeline_validate_strict_raises_on_cycle(registry: BlockRegistry) -> None:
    g1, g2 = _make_two_graphs(registry)
    p = Pipeline()
    p.add_graph(g1, "G1")
    p.add_graph(g2, "G2")
    p.add_edge("G1", "result", "G2", "a_in")
    p.add_edge("G2", "result", "G1", "a_in")
    p.expose_input("G1", "b_in", name="b")
    p.expose_output("G2", "result", name="out")
    with pytest.raises(ValueError, match="cycle"):
        p.validate(strict=True)


def test_pipeline_validate_reachability_warnings(registry: BlockRegistry) -> None:
    g1, g2 = _make_two_graphs(registry)
    p = Pipeline()
    p.add_graph(g1, "G1")
    p.add_graph(g2, "G2")
    p.add_edge("G1", "result", "G2", "a_in")
    p.expose_input("G1", "a_in", name="a")
    p.expose_input("G1", "b_in", name="b")
    p.expose_output("G2", "result", name="out")
    result = p.validate(strict=False)
    assert result.is_valid
    # G1 and G2 are reachable from inputs and lead to output; no warning expected
    assert not [w for w in result.warnings if "reachable" in w.lower() or "lead" in w.lower()]


# --- run ---


def test_pipeline_run_single_graph(registry: BlockRegistry) -> None:
    g = _make_single_graph(registry, offset=2)
    p = Pipeline()
    p.add_graph(g, "G1")
    p.expose_input("G1", "a_in", name="a")
    p.expose_input("G1", "b_in", name="b")
    p.expose_output("G1", "result", name="out")
    out = p.run({"a": 1, "b": 3})
    assert out == {"out": 6}  # 1+3+2


def test_pipeline_run_two_graphs(registry: BlockRegistry) -> None:
    g1, g2 = _make_two_graphs(registry)
    p = Pipeline()
    p.add_graph(g1, "G1")
    p.add_graph(g2, "G2")
    p.add_edge("G1", "result", "G2", "a_in")
    p.expose_input("G1", "a_in", name="a")
    p.expose_input("G1", "b_in", name="b")
    p.expose_output("G2", "result", name="out")
    out = p.run({"a": 3, "b": 5})
    # G1: 3+5=8 -> G2.a_in=8, G2.b_in not set -> 0, so 8+10=18
    assert out == {"out": 18}


def test_pipeline_run_with_callbacks(registry: BlockRegistry) -> None:
    g = _make_single_graph(registry)
    p = Pipeline()
    p.add_graph(g, "G1")
    p.expose_input("G1", "a_in", name="a")
    p.expose_input("G1", "b_in", name="b")
    p.expose_output("G1", "result", name="out")
    seen = []

    def hook(node_id, phase, **kwargs):
        seen.append((node_id, phase))

    out = p.run({"a": 1, "b": 2}, callbacks=[hook])
    assert out["out"] == 3
    assert ("G1", "before") in seen
    assert ("G1", "after") in seen


# --- get_input_spec / get_output_spec ---


def test_pipeline_get_input_output_spec(registry: BlockRegistry) -> None:
    g = _make_single_graph(registry)
    p = Pipeline()
    p.add_graph(g, "G1")
    p.expose_input("G1", "a_in", name="a")
    p.expose_input("G1", "b_in", name="b")
    p.expose_output("G1", "result", name="out")
    ins = p.get_input_spec()
    outs = p.get_output_spec()
    keys_in = [e.get("name") or f"{e['pipeline_node_id']}:{e['port_name']}" for e in ins]
    keys_out = [e.get("name") or f"{e['pipeline_node_id']}:{e['port_name']}" for e in outs]
    assert "a" in keys_in and "b" in keys_in
    assert "out" in keys_out


# --- to_config / from_config ---


def test_pipeline_from_yaml(registry: BlockRegistry, tmp_path: Path) -> None:
    g1 = _make_single_graph(registry, offset=2)
    cfg = {
        "pipeline_id": "p1",
        "graphs": [{"pipeline_node_id": "G1", "graph_config": g1.to_config()}],
        "edges": [],
        "exposed_inputs": [
            {"pipeline_node_id": "G1", "port_name": "a_in", "name": "a"},
            {"pipeline_node_id": "G1", "port_name": "b_in", "name": "b"},
        ],
        "exposed_outputs": [{"pipeline_node_id": "G1", "port_name": "result", "name": "out"}],
    }
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    p = Pipeline.from_yaml(str(path), registry=registry)
    out = p.run({"a": 1, "b": 3})
    assert out["out"] == 6


def test_pipeline_to_config_from_config_roundtrip(registry: BlockRegistry) -> None:
    g1, g2 = _make_two_graphs(registry)
    p = Pipeline(pipeline_id="p1")
    p.add_graph(g1, "G1")
    p.add_graph(g2, "G2")
    p.add_edge("G1", "result", "G2", "a_in")
    p.expose_input("G1", "a_in", name="a")
    p.expose_input("G1", "b_in", name="b")
    p.expose_output("G2", "result", name="out")
    cfg = p.to_config()
    assert "graphs" in cfg
    assert "edges" in cfg
    p2 = Pipeline.from_config(cfg, registry=registry)
    assert p2.node_ids == p.node_ids
    out = p2.run({"a": 2, "b": 4})
    assert out["out"] == 16  # 2+4=6 -> G2.a_in=6, b_in=0 -> 6+10=16


# --- save / load ---


def test_pipeline_save_and_load_directory(registry: BlockRegistry, tmp_path: Path) -> None:
    g = _make_single_graph(registry, offset=7)
    p = Pipeline()
    p.add_graph(g, "G1")
    p.expose_input("G1", "a_in", name="a")
    p.expose_input("G1", "b_in", name="b")
    p.expose_output("G1", "result", name="out")
    save_dir = str(tmp_path / "pipe")
    p.save(save_dir)
    assert (tmp_path / "pipe" / "config.json").exists()
    assert (tmp_path / "pipe" / "checkpoints" / "G1" / "checkpoint.json").exists()
    p2 = Pipeline.load(save_dir, registry=registry)
    out = p2.run({"a": 1, "b": 2})
    assert out["out"] == 10  # 1+2+7


# --- trainable ---


def test_pipeline_trainable_parameters_and_set_trainable(registry: BlockRegistry) -> None:
    g = _make_single_graph(registry)
    p = Pipeline()
    p.add_graph(g, "G1")
    params = list(p.trainable_parameters())
    p.set_trainable("G1", False)
    params_off = list(p.trainable_parameters())
    assert len(params_off) <= len(params)


# --- infer_exposed_ports ---


def test_pipeline_infer_exposed_ports(registry: BlockRegistry) -> None:
    g1, g2 = _make_two_graphs(registry)
    p = Pipeline()
    p.add_graph(g1, "G1")
    p.add_graph(g2, "G2")
    p.add_edge("G1", "result", "G2", "a_in")
    p.infer_exposed_ports()
    ins = p.get_input_spec()
    outs = p.get_output_spec()
    in_keys = [e.get("name") or f"{e['pipeline_node_id']}:{e['port_name']}" for e in ins]
    out_keys = [e.get("name") or f"{e['pipeline_node_id']}:{e['port_name']}" for e in outs]
    assert "G1:a_in" in in_keys or "a_in" in in_keys
    assert "G1:b_in" in in_keys or "b_in" in in_keys
    assert "G2:result" in out_keys or "result" in out_keys


# --- PipelineEdge ---


def test_pipeline_edge_empty_string_raises() -> None:
    with pytest.raises(ValueError):
        PipelineEdge("", "out", "G2", "in")
    with pytest.raises(ValueError):
        PipelineEdge("G1", "out", "G2", "")
