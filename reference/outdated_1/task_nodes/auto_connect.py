"""
Auto-connect new nodes by role rules (suggest_edges_for_new_node + add_edge).

Canon: WorldGenerator_2.0/TODO_02 B.2.2. Foundation does not depend on task_nodes;
the graph's auto_connect_fn callback is set by use_task_node_auto_connect(graph).
"""

from __future__ import annotations

from typing import Any

from yggdrasill.foundation.graph import Edge, Graph
from yggdrasill.task_nodes.role_rules import suggest_edges_for_new_node
from yggdrasill.task_nodes.roles import role_from_block_type


def apply_auto_connect(graph: Graph, new_node_id: str, new_block: Any) -> None:
    """
    Add edges suggested by role rules for the new node. Skips edges whose ports
    are missing or incompatible (concrete blocks may not implement all canonical ports).
    """
    new_role = role_from_block_type(getattr(new_block, "block_type", "") or "")
    if not new_role:
        return
    existing: list[tuple[str, str]] = []
    for nid in graph.node_ids:
        if nid == new_node_id:
            continue
        node = graph.get_node(nid)
        if not node:
            continue
        role = role_from_block_type(getattr(node.block, "block_type", "") or "")
        if role:
            existing.append((nid, role))
    for src_nid, src_port, tgt_nid, tgt_port in suggest_edges_for_new_node(
        new_node_id, new_role, existing
    ):
        if _edge_exists(graph, src_nid, src_port, tgt_nid, tgt_port):
            continue
        if not _ports_ok(graph, src_nid, src_port, tgt_nid, tgt_port):
            continue
        try:
            graph.add_edge(Edge(src_nid, src_port, tgt_nid, tgt_port))
        except ValueError:
            pass


def _edge_exists(
    graph: Graph, src_nid: str, src_port: str, tgt_nid: str, tgt_port: str
) -> bool:
    for e in graph.get_edges_out(src_nid):
        if e.source_port == src_port and e.target_node == tgt_nid and e.target_port == tgt_port:
            return True
    return False


def _ports_ok(
    graph: Graph, src_nid: str, src_port: str, tgt_nid: str, tgt_port: str
) -> bool:
    sn = graph.get_node(src_nid)
    tn = graph.get_node(tgt_nid)
    if not sn or not tn:
        return False
    sp = sn.block.get_port(src_port)
    tp = tn.block.get_port(tgt_port)
    if sp is None or tp is None:
        return False
    return sp.compatible_with(tp)


def use_task_node_auto_connect(graph: Graph) -> None:
    """Set graph.auto_connect_fn so add_node(..., auto_connect=True) uses role rules."""
    graph.auto_connect_fn = apply_auto_connect
