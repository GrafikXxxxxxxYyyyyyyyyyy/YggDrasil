"""Auto-connect: automatically create edges when adding a task-node.

Based on PHASE_4 §9 and Canon 02 §12.2.
"""
from __future__ import annotations

from typing import Any, Dict

from yggdrasill.engine.edge import Edge
from yggdrasill.foundation.node import AbstractGraphNode
from yggdrasill.task_nodes.role_rules import suggest_edges_for_new_node
from yggdrasill.task_nodes.roles import role_from_block_type


def apply_auto_connect(
    hypergraph: Any,
    new_node_id: str,
    new_block: Any,
) -> int:
    """Create edges for *new_node_id* based on role rules.

    Returns the number of edges actually added.
    """
    bt = getattr(new_block, "block_type", None)
    if bt is None:
        return 0
    new_role = role_from_block_type(bt)
    if new_role is None:
        return 0

    existing: Dict[str, str] = {}
    for nid in hypergraph.node_ids:
        if nid == new_node_id:
            continue
        node = hypergraph.get_node(nid)
        node_bt = getattr(node, "block_type", None)
        if node_bt is None:
            continue
        r = role_from_block_type(node_bt)
        if r is not None:
            existing[nid] = r

    suggestions = suggest_edges_for_new_node(new_node_id, new_role, existing)
    added = 0
    for src_nid, src_port, tgt_nid, tgt_port in suggestions:
        src_node = hypergraph.get_node(src_nid)
        tgt_node = hypergraph.get_node(tgt_nid)
        src_port_obj = None
        tgt_port_obj = None
        if isinstance(src_node, AbstractGraphNode):
            src_port_obj = src_node.get_port(src_port)
            if src_port_obj is None:
                continue
        if isinstance(tgt_node, AbstractGraphNode):
            tgt_port_obj = tgt_node.get_port(tgt_port)
            if tgt_port_obj is None:
                continue
        if src_port_obj is not None and tgt_port_obj is not None:
            if not src_port_obj.compatible_with(tgt_port_obj):
                continue
        edge = Edge(src_nid, src_port, tgt_nid, tgt_port)
        try:
            hypergraph.add_edge(edge)
            added += 1
        except (ValueError, KeyError):
            pass
    return added


def use_task_node_auto_connect(hypergraph: Any) -> None:
    """Enable auto-connect on *hypergraph*.

    After calling this, ``hypergraph.add_node_from_config(..., auto_connect=True)``
    will automatically create edges based on role rules.

    Implementation note: sets ``_auto_connect_fn`` on the hypergraph so that
    ``add_node_from_config`` can invoke it when ``auto_connect=True`` is passed
    via **kwargs.
    """
    hypergraph._auto_connect_fn = apply_auto_connect
