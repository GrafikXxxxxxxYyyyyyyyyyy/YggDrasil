"""Canonical connection rules between task-node roles.

Based on Canon 02 sections 4.5--10.5: "typical connections" for each role pair.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from yggdrasill.task_nodes.roles import (
    BACKBONE, CONJECTOR, CONVERTER, HELPER,
    INJECTOR, INNER_MODULE, OUTER_MODULE,
)

RolePair = Tuple[str, str]
PortPair = Tuple[str, str]

ROLE_EDGE_RULES: Dict[RolePair, List[PortPair]] = {
    # Backbone interactions
    (CONJECTOR, BACKBONE): [("condition", "condition")],
    (INJECTOR, BACKBONE): [("adapted", "condition")],
    (CONVERTER, BACKBONE): [("output", "latent")],
    (OUTER_MODULE, BACKBONE): [("output", "latent")],
    (HELPER, BACKBONE): [("result", "latent")],
    (BACKBONE, CONVERTER): [("pred", "input")],
    (BACKBONE, INNER_MODULE): [("pred", "pred")],
    # Inner module interactions
    (INNER_MODULE, BACKBONE): [("next_latent", "latent"), ("next_timestep", "timestep")],
    (OUTER_MODULE, INNER_MODULE): [("output", "latent")],
    (INJECTOR, INNER_MODULE): [("adapted", "control")],
    # Converter / Conjector chains
    (CONVERTER, CONJECTOR): [("output", "input")],
    (CONVERTER, CONVERTER): [("output", "input")],
    (CONJECTOR, INJECTOR): [("condition", "condition")],
    # Outer module chains
    (OUTER_MODULE, OUTER_MODULE): [("output", "input")],
    # Helper chains
    (HELPER, CONVERTER): [("result", "input")],
    (HELPER, HELPER): [("result", "input")],
}


def get_rule_edges(
    source_role: str,
    target_role: str,
) -> List[PortPair]:
    """Return the canonical (source_port, target_port) pairs for a role combination."""
    return list(ROLE_EDGE_RULES.get((source_role, target_role), []))


def suggest_edges_for_new_node(
    new_node_id: str,
    new_role: str,
    existing_nodes: Dict[str, str],
) -> List[Tuple[str, str, str, str]]:
    """Suggest edges for *new_node_id* (with *new_role*) against existing nodes.

    Parameters
    ----------
    new_node_id : str
        ID of the newly added node.
    new_role : str
        Canonical role string of the new node.
    existing_nodes : dict
        Mapping ``{node_id: role_string}`` for all other nodes already in the graph.

    Returns
    -------
    list of (source_node, source_port, target_node, target_port) tuples.
    """
    suggestions: List[Tuple[str, str, str, str]] = []
    for ex_id, ex_role in existing_nodes.items():
        for src_port, tgt_port in get_rule_edges(new_role, ex_role):
            suggestions.append((new_node_id, src_port, ex_id, tgt_port))
        for src_port, tgt_port in get_rule_edges(ex_role, new_role):
            suggestions.append((ex_id, src_port, new_node_id, tgt_port))
    return suggestions
