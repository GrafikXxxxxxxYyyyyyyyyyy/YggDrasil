"""
Role rules: default port connections for auto_connect (source_role -> target_role).

Canon: WorldGenerator_2.0/TODO_02 A.10, Abstract_Task_Nodes.md §3–9.
Each rule: (source_role, target_role) -> [(source_port, target_port), ...].
When adding a new node with role R_new, for each existing node with role R_existing:
- (R_existing, R_new): edge from existing -> new: (existing_node, src_port) -> (new_node, dst_port)
- (R_new, R_existing): edge from new -> existing: (new_node, src_port) -> (existing_node, dst_port)
"""

from __future__ import annotations

from typing import List, Tuple

from yggdrasill.task_nodes.roles import (
    ROLE_BACKBONE,
    ROLE_SOLVER,
    ROLE_CODEC,
    ROLE_CONDITIONER,
    ROLE_TOKENIZER,
    ROLE_ADAPTER,
    ROLE_GUIDANCE,
)

# (source_role, target_role) -> [(source_port, target_port), ...]
_ROLE_RULES: List[Tuple[str, str, List[Tuple[str, str]]]] = [
    # tokenizer -> conditioner
    (ROLE_TOKENIZER, ROLE_CONDITIONER, [("token_ids", "input")]),
    # conditioner -> backbone
    (ROLE_CONDITIONER, ROLE_BACKBONE, [("embedding", "condition")]),
    # backbone -> solver
    (ROLE_BACKBONE, ROLE_SOLVER, [("pred", "pred")]),
    # solver -> backbone (cycle)
    (ROLE_SOLVER, ROLE_BACKBONE, [("next_latent", "latent"), ("next_timestep", "timestep")]),
    # codec encode output -> solver (initial latent)
    (ROLE_CODEC, ROLE_SOLVER, [("encode_latent", "latent")]),
    # solver -> codec decode (final latent)
    (ROLE_SOLVER, ROLE_CODEC, [("next_latent", "decode_latent")]),
    # conditioner -> adapter
    (ROLE_CONDITIONER, ROLE_ADAPTER, [("embedding", "condition")]),
    # adapter -> backbone (adapted as condition; aggregation if multiple)
    (ROLE_ADAPTER, ROLE_BACKBONE, [("adapted", "condition")]),
    # backbone -> guidance (conditional pred)
    (ROLE_BACKBONE, ROLE_GUIDANCE, [("pred", "pred_cond")]),
    # guidance -> solver
    (ROLE_GUIDANCE, ROLE_SOLVER, [("pred_guided", "pred")]),
]

_rules_map: dict[Tuple[str, str], List[Tuple[str, str]]] | None = None


def _build_rules_map() -> dict[Tuple[str, str], List[Tuple[str, str]]]:
    global _rules_map
    if _rules_map is None:
        _rules_map = {}
        for src_role, tgt_role, pairs in _ROLE_RULES:
            _rules_map[(src_role, tgt_role)] = pairs
    return _rules_map


def get_rule_edges(source_role: str, target_role: str) -> List[Tuple[str, str]]:
    """
    Return [(source_port, target_port), ...] for edges from source_role node to target_role node.
    """
    return list(_build_rules_map().get((source_role, target_role), []))


def suggest_edges_for_new_node(
    new_node_id: str,
    new_role: str,
    existing_roles_by_node_id: List[Tuple[str, str]],
) -> List[Tuple[str, str, str, str]]:
    """
    Suggest edges when a new node (new_node_id, new_role) is added.
    existing_roles_by_node_id: [(node_id, role), ...].
    Returns [(source_node_id, source_port, target_node_id, target_port), ...].
    """
    out: List[Tuple[str, str, str, str]] = []
    for existing_node_id, existing_role in existing_roles_by_node_id:
        for src_port, dst_port in get_rule_edges(existing_role, new_role):
            out.append((existing_node_id, src_port, new_node_id, dst_port))
        for src_port, dst_port in get_rule_edges(new_role, existing_role):
            out.append((new_node_id, src_port, existing_node_id, dst_port))
    return out