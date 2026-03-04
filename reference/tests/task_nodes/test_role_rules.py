"""Tests for task_nodes.role_rules."""

from yggdrasill.task_nodes.role_rules import (
    get_rule_edges,
    suggest_edges_for_new_node,
)
from yggdrasill.task_nodes.roles import (
    ROLE_BACKBONE,
    ROLE_SOLVER,
    ROLE_CONDITIONER,
    ROLE_CODEC,
)


def test_get_rule_edges_backbone_solver() -> None:
    edges = get_rule_edges(ROLE_BACKBONE, ROLE_SOLVER)
    assert ("pred", "pred") in edges


def test_get_rule_edges_solver_backbone() -> None:
    edges = get_rule_edges(ROLE_SOLVER, ROLE_BACKBONE)
    assert ("next_latent", "latent") in edges
    assert ("next_timestep", "timestep") in edges


def test_get_rule_edges_conditioner_backbone() -> None:
    edges = get_rule_edges(ROLE_CONDITIONER, ROLE_BACKBONE)
    assert ("embedding", "condition") in edges


def test_get_rule_edges_codec_solver() -> None:
    edges = get_rule_edges(ROLE_CODEC, ROLE_SOLVER)
    assert ("encode_latent", "latent") in edges


def test_suggest_edges_for_new_node() -> None:
    existing = [("cond", "conditioner"), ("bb", "backbone")]
    suggested = suggest_edges_for_new_node("solver_0", "solver", existing)
    # solver new: (backbone -> solver) and (solver -> backbone), (conditioner not in rule with solver)
    # backbone->solver: pred->pred; solver->backbone: next_latent->latent, next_timestep->timestep
    src_tgt = {(s, sp, t, tp) for s, sp, t, tp in suggested}
    assert ("bb", "pred", "solver_0", "pred") in src_tgt
    assert ("solver_0", "next_latent", "bb", "latent") in src_tgt
    assert ("solver_0", "next_timestep", "bb", "timestep") in src_tgt
