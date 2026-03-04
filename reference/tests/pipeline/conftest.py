"""Shared fixtures for pipeline tests."""

import pytest

from yggdrasill.foundation.registry import BlockRegistry
from tests.foundation.helpers import AddBlock, IdentityBlock


@pytest.fixture
def registry() -> BlockRegistry:
    r = BlockRegistry()
    r.register("add", AddBlock)
    r.register("identity", IdentityBlock)
    return r
