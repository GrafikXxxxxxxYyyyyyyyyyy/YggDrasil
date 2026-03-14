"""Test helpers for the engine layer -- reuses foundation helpers."""
from __future__ import annotations

from yggdrasill.engine.edge import Edge
from yggdrasill.engine.structure import Hypergraph
from tests.foundation.helpers import IdentityTaskNode


def make_chain(*names: str) -> Hypergraph:
    """Build a linear chain: A -> B -> C ...  using IdentityTaskNode (in -> out)."""
    h = Hypergraph()
    for name in names:
        h.add_node(name, IdentityTaskNode(node_id=name))
    for i in range(len(names) - 1):
        h.add_edge(Edge(names[i], "out", names[i + 1], "in"))
    if names:
        h.expose_input(names[0], "in", "x")
        h.expose_output(names[-1], "out", "y")
    return h


def make_cycle(a: str = "A", b: str = "B") -> Hypergraph:
    """Build a two-node cycle A <-> B with IdentityTaskNode."""
    h = Hypergraph()
    h.add_node(a, IdentityTaskNode(node_id=a))
    h.add_node(b, IdentityTaskNode(node_id=b))
    h.add_edge(Edge(a, "out", b, "in"))
    h.add_edge(Edge(b, "out", a, "in"))
    h.expose_input(a, "in", "x")
    h.expose_output(b, "out", "y")
    h.metadata = {"num_loop_steps": 2}
    return h
