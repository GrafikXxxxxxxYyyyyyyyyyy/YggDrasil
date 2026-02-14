"""Tests for assembler classes."""
import pytest

from yggdrasil.core.block.registry import auto_discover


@pytest.fixture(autouse=True)
def discover():
    auto_discover()


class TestModelAssembler:
    def test_from_config_dict(self):
        """Test building a model from a plain dict config (with identity codec)."""
        from yggdrasil.assemblers.model_assembler import ModelAssembler
        
        config = {
            "model": {
                "type": "model/modular",
                "codec": {"type": "codec/identity", "latent_channels": 4},
            }
        }
        model = ModelAssembler.from_config(config)
        assert model is not None
        assert model.block_type == "model/modular"


class TestPipelineAssembler:
    def test_for_generation(self):
        """Test quick assembly of a generation pipeline."""
        from yggdrasil.assemblers.pipeline_assembler import PipelineAssembler
        from yggdrasil.assemblers.model_assembler import ModelAssembler
        
        config = {
            "type": "model/modular",
            "codec": {"type": "codec/identity"},
        }
        model = ModelAssembler.from_config(config)
        
        sampler = PipelineAssembler.for_generation(
            model=model,
            num_steps=5,
            guidance_scale=1.0,
        )
        assert sampler is not None


class TestAdapterAssembler:
    def test_list_adapters(self):
        """Test listing adapters on a model."""
        from yggdrasil.assemblers.adapter_assembler import AdapterAssembler
        from yggdrasil.assemblers.model_assembler import ModelAssembler
        
        config = {
            "type": "model/modular",
            "codec": {"type": "codec/identity"},
        }
        model = ModelAssembler.from_config(config)
        adapters = AdapterAssembler.list_adapters(model)
        assert isinstance(adapters, list)


class TestMultiModalAssembler:
    def test_create_empty(self):
        """Test creating an empty multi-modal pipeline."""
        from yggdrasil.assemblers.multi_modal_assembler import MultiModalAssembler
        
        assembler = MultiModalAssembler()
        assert len(assembler) == 0
    
    def test_repr(self):
        from yggdrasil.assemblers.multi_modal_assembler import MultiModalAssembler
        assembler = MultiModalAssembler()
        assert "MultiModalPipeline" in repr(assembler)
