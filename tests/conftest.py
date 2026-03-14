import pytest
from yggdrasill.foundation.registry import BlockRegistry


@pytest.fixture(autouse=True)
def _preserve_global_registry():
    """Save and restore the global registry (both object ref and factories) around each test."""
    old_global = BlockRegistry._global
    reg = BlockRegistry.global_registry()
    snapshot = dict(reg._factories)
    yield
    BlockRegistry._global = old_global
    if old_global is not None:
        old_global._factories.clear()
        old_global._factories.update(snapshot)
