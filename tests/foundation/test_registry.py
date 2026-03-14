import pytest

from yggdrasill.foundation.registry import BlockRegistry, register_block
from tests.foundation.helpers import AddBlock, IdentityBlock, IdentityTaskNode


@pytest.fixture()
def registry():
    """Fresh local registry for each test."""
    r = BlockRegistry()
    r.register("test/identity", IdentityBlock)
    r.register("test/add", AddBlock)
    r.register("test/identity_task", IdentityTaskNode)
    return r


class TestRegistryRegisterAndBuild:
    def test_build_identity(self, registry: BlockRegistry):
        block = registry.build({"block_type": "test/identity"})
        assert block.block_type == "test/identity"

    def test_build_add_with_config(self, registry: BlockRegistry):
        block = registry.build({
            "block_type": "test/add",
            "block_id": "a1",
            "offset": 5,
        })
        assert block.block_id == "a1"
        assert block.forward({"a": 1, "b": 2}) == {"out": 8}

    def test_build_with_type_key(self, registry: BlockRegistry):
        registry.register("alt", IdentityBlock)
        block = registry.build({"type": "alt"})
        assert block.block_type == "test/identity"

    def test_build_task_node(self, registry: BlockRegistry):
        node = registry.build({
            "block_type": "test/identity_task",
            "node_id": "N1",
            "block_id": "b1",
        })
        assert hasattr(node, "node_id")
        assert node.node_id == "N1"  # type: ignore[attr-defined]
        assert node.block_id == "b1"
        assert node.run({"in": 7}) == {"out": 7}  # type: ignore[attr-defined]


class TestRegistryErrors:
    def test_missing_block_type(self, registry: BlockRegistry):
        with pytest.raises(KeyError, match="block_type"):
            registry.build({"foo": "bar"})

    def test_unknown_block_type(self, registry: BlockRegistry):
        with pytest.raises(KeyError, match="Unknown block_type"):
            registry.build({"block_type": "nonexistent"})


class TestRegistryLookup:
    def test_get(self, registry: BlockRegistry):
        assert registry.get("test/identity") is IdentityBlock

    def test_get_missing(self, registry: BlockRegistry):
        assert registry.get("nope") is None

    def test_contains(self, registry: BlockRegistry):
        assert "test/identity" in registry
        assert "nope" not in registry


class TestRegisterBlockDecorator:
    def test_decorator(self):
        r = BlockRegistry()

        @register_block("my/block", registry=r)
        class MyBlock(IdentityBlock):
            pass

        assert "my/block" in r
        b = r.build({"block_type": "my/block"})
        assert isinstance(b, MyBlock)


class TestGlobalRegistry:
    def test_global_registry_singleton(self):
        BlockRegistry.reset_global()
        r1 = BlockRegistry.global_registry()
        r2 = BlockRegistry.global_registry()
        assert r1 is r2
        BlockRegistry.reset_global()

    def test_local_does_not_affect_global(self, registry: BlockRegistry):
        BlockRegistry.reset_global()
        g = BlockRegistry.global_registry()
        assert "test/identity" not in g
        assert "test/identity" in registry
        BlockRegistry.reset_global()
