"""Auto-connect for workflows: match exposed ports between hypergraphs.

PHASE_6 §13.2: suggest_auto_edges and apply_auto_connect at workflow level.
"""
from __future__ import annotations

from typing import Any, List, Tuple


def suggest_auto_edges(
    workflow: Any,
) -> List[Tuple[str, str, str, str]]:
    """Suggest edges between hypergraphs by matching port names and dtypes.

    For every pair (A, B) of graphs in the workflow, if A has an exposed output
    with the same ``port_name`` (and compatible dtype) as B's exposed input, and
    no workflow edge already covers that connection, suggest it.

    Returns list of ``(source_graph_id, source_port, target_graph_id, target_port)``.
    """
    suggestions: List[Tuple[str, str, str, str]] = []
    existing_edges = {
        (e.source_node, e.source_port, e.target_node, e.target_port)
        for e in workflow.get_edges()
    }

    graph_ids = list(workflow.node_ids)
    out_specs = {}
    in_specs = {}
    for gid in graph_ids:
        g = workflow.get_node(gid)
        out_specs[gid] = g.get_output_spec(include_dtype=True)
        in_specs[gid] = g.get_input_spec(include_dtype=True)

    for src_gid in graph_ids:
        for dst_gid in graph_ids:
            if src_gid == dst_gid:
                continue
            for o_entry in out_specs[src_gid]:
                o_name = o_entry["port_name"]
                o_dtype = o_entry.get("dtype", "any")
                for i_entry in in_specs[dst_gid]:
                    i_name = i_entry["port_name"]
                    i_dtype = i_entry.get("dtype", "any")
                    if o_name != i_name:
                        continue
                    if o_dtype != "any" and i_dtype != "any" and o_dtype != i_dtype:
                        continue
                    candidate = (src_gid, o_name, dst_gid, i_name)
                    if candidate not in existing_edges:
                        suggestions.append(candidate)

    return suggestions


def apply_auto_connect(workflow: Any) -> int:
    """Create all suggested edges. Returns count of edges added."""
    added = 0
    for src_g, src_p, dst_g, dst_p in suggest_auto_edges(workflow):
        try:
            workflow.add_edge(src_g, src_p, dst_g, dst_p)
            added += 1
        except (ValueError, KeyError):
            pass
    return added
