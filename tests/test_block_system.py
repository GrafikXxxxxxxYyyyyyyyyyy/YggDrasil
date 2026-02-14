"""Tests for the block system: AbstractBlock, Slot, Builder, Registry."""
import pytest
import torch
from omegaconf import OmegaConf

from yggdrasil.core.block.base import AbstractBlock
from yggdrasil.core.block.slot import Slot
from yggdrasil.core.block.registry import BlockRegistry, register_block, auto_discover
from yggdrasil.core.block.builder import BlockBuilder


# === Test AbstractBlock ===

class TestAbstractBlock:
    def test_block_has_id(self):
        """Each block should get a unique ID."""
        auto_discover()
        
        @register_block("test/dummy_a")
        class DummyBlockA(AbstractBlock):
            block_type = "test/dummy_a"
            def _forward_impl(self, *args, **kwargs):
                return None
        
        block = DummyBlockA({"type": "test/dummy_a"})
        assert block.block_id is not None
        assert len(block.block_id) > 0
    
    def test_block_has_config(self):
        auto_discover()
        
        @register_block("test/dummy_b")
        class DummyBlockB(AbstractBlock):
            block_type = "test/dummy_b"
            def _forward_impl(self, *args, **kwargs):
                return None
        
        config = {"type": "test/dummy_b", "param1": 42}
        block = DummyBlockB(config)
        assert block.config.get("param1") == 42


# === Test Slot ===

class TestSlot:
    def test_slot_check_compatible(self):
        slot = Slot(name="test", accepts=AbstractBlock, multiple=False)
        
        @register_block("test/compat")
        class CompatBlock(AbstractBlock):
            block_type = "test/compat"
            def _forward_impl(self, *args, **kwargs):
                return None
        
        block = CompatBlock({"type": "test/compat"})
        assert slot.check_compatible(block) is True
    
    def test_slot_rejects_class(self):
        slot = Slot(name="test", accepts=AbstractBlock, multiple=False)
        assert slot.check_compatible(AbstractBlock) is False
    
    def test_slot_string_accepts(self):
        slot = Slot(name="test", accepts="backbone/", multiple=False)
        
        @register_block("backbone/test_str")
        class StrBlock(AbstractBlock):
            block_type = "backbone/test_str"
            def _forward_impl(self, *args, **kwargs):
                return None
        
        block = StrBlock({"type": "backbone/test_str"})
        assert slot.check_compatible(block) is True


# === Test Registry ===

class TestRegistry:
    def test_register_and_get(self):
        @register_block("test/registry_test")
        class RegTestBlock(AbstractBlock):
            block_type = "test/registry_test"
            def _forward_impl(self, *args, **kwargs):
                return None
        
        cls = BlockRegistry.get("test/registry_test")
        assert cls is RegTestBlock
    
    def test_list_blocks(self):
        auto_discover()
        blocks = BlockRegistry.list_blocks()
        assert len(blocks) > 0
    
    def test_missing_block_raises(self):
        with pytest.raises(KeyError):
            BlockRegistry.get("nonexistent/block_xyz_123")


# === Test Builder ===

class TestBuilder:
    def test_build_from_dict(self):
        auto_discover()
        
        @register_block("test/buildable")
        class BuildableBlock(AbstractBlock):
            block_type = "test/buildable"
            def _forward_impl(self, *args, **kwargs):
                return None
        
        block = BlockBuilder.build({"type": "test/buildable", "value": 99})
        assert isinstance(block, BuildableBlock)
    
    def test_build_from_omegaconf(self):
        auto_discover()
        
        @register_block("test/buildable_oc")
        class BuildableOCBlock(AbstractBlock):
            block_type = "test/buildable_oc"
            def _forward_impl(self, *args, **kwargs):
                return None
        
        config = OmegaConf.create({"type": "test/buildable_oc"})
        block = BlockBuilder.build(config)
        assert isinstance(block, BuildableOCBlock)
