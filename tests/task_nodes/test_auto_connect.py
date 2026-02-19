"""Tests for task_nodes.auto_connect and add_node(..., auto_connect=True)."""

from yggdrasill.foundation.graph import Graph
from yggdrasill.task_nodes import use_task_node_auto_connect
from yggdrasill.task_nodes.stubs import (
    IdentityBackbone,
    IdentitySolver,
    IdentityConditioner,
)
from yggdrasill.task_nodes import stubs  # ensure stubs registered


def test_auto_connect_adds_edges_between_backbone_and_solver() -> None:
    reg = stubs  # use module so decorators have run
    from yggdrasill.foundation.registry import BlockRegistry
    registry = BlockRegistry.global_registry()
    g = Graph()
    use_task_node_auto_connect(g)
    g.add_node("bb", "backbone/identity", registry=registry, auto_connect=True)
    g.add_node("sol", "solver/identity", registry=registry, auto_connect=True)
    edges = g.get_edges()
    # backbone pred -> solver pred; solver next_latent -> backbone latent, next_timestep -> backbone timestep
    edge_set = {(e.source_node, e.source_port, e.target_node, e.target_port) for e in edges}
    assert ("bb", "pred", "sol", "pred") in edge_set
    assert ("sol", "next_latent", "bb", "latent") in edge_set
    assert ("sol", "next_timestep", "bb", "timestep") in edge_set


def test_auto_connect_conditioner_to_backbone() -> None:
    from yggdrasill.foundation.registry import BlockRegistry
    registry = BlockRegistry.global_registry()
    g = Graph()
    use_task_node_auto_connect(g)
    g.add_node("cond", "conditioner/identity", registry=registry, auto_connect=True)
    g.add_node("bb", "backbone/identity", registry=registry, auto_connect=True)
    edges = g.get_edges()
    edge_set = {(e.source_node, e.source_port, e.target_node, e.target_port) for e in edges}
    assert ("cond", "embedding", "bb", "condition") in edge_set
