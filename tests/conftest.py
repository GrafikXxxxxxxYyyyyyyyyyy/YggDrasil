import pytest
from yggdrasill.foundation.registry import BlockRegistry


@pytest.fixture(autouse=True)
def _preserve_global_registry():
    """Save and restore the global registry around each test."""
    old = BlockRegistry._global
    yield
    BlockRegistry._global = old
