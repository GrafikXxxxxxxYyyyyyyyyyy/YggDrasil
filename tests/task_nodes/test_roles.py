"""Tests for task_nodes.roles."""

import pytest

from yggdrasill.task_nodes.roles import (
    KNOWN_ROLES,
    ROLE_BACKBONE,
    ROLE_SOLVER,
    ROLE_CODEC,
    role_from_block_type,
)


def test_known_roles() -> None:
    assert ROLE_BACKBONE in KNOWN_ROLES
    assert ROLE_SOLVER in KNOWN_ROLES
    assert ROLE_CODEC in KNOWN_ROLES
    assert len(KNOWN_ROLES) >= 7


def test_role_from_block_type_prefix() -> None:
    assert role_from_block_type("backbone/unet2d") == "backbone"
    assert role_from_block_type("solver/ddim") == "solver"
    assert role_from_block_type("codec/vae") == "codec"
    assert role_from_block_type("conditioner/identity") == "conditioner"
    assert role_from_block_type("tokenizer/bpe") == "tokenizer"
    assert role_from_block_type("adapter/lora") == "adapter"
    assert role_from_block_type("guidance/cfg") == "guidance"


def test_role_from_block_type_bare() -> None:
    assert role_from_block_type("backbone") == "backbone"
    assert role_from_block_type("solver") == "solver"


def test_role_from_block_type_unknown() -> None:
    assert role_from_block_type("unknown/foo") is None
    assert role_from_block_type("") is None
