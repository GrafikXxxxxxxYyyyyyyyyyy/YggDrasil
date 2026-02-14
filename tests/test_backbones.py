"""Tests for backbone blocks."""
import pytest
import torch

from yggdrasil.core.block.registry import auto_discover


@pytest.fixture(autouse=True)
def discover():
    auto_discover()


class TestDiTBackbone:
    def test_forward(self):
        from yggdrasil.blocks.backbones.dit import DiTBackbone
        
        backbone = DiTBackbone({
            "type": "backbone/dit",
            "hidden_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "patch_size": 2,
            "in_channels": 4,
            "cond_dim": 64,
        })
        
        x = torch.randn(1, 4, 8, 8)
        t = torch.tensor([500])
        condition = {"encoder_hidden_states": torch.randn(1, 10, 64)}
        
        output = backbone._forward_impl(x, t, condition)
        assert output.shape == x.shape


class TestMMDiTBackbone:
    def test_forward(self):
        from yggdrasil.blocks.backbones.mmdit import MMDiTBackbone
        
        backbone = MMDiTBackbone({
            "type": "backbone/mmdit",
            "hidden_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "patch_size": 2,
            "in_channels": 4,
            "cond_dim": 64,
        })
        
        x = torch.randn(1, 4, 8, 8)
        t = torch.tensor([500])
        condition = {"encoder_hidden_states": torch.randn(1, 10, 64)}
        
        output = backbone._forward_impl(x, t, condition)
        assert output.shape == x.shape


class TestTransformer1DBackbone:
    def test_forward(self):
        from yggdrasil.blocks.backbones.transformer_1d import Transformer1DBackbone
        
        backbone = Transformer1DBackbone({
            "type": "backbone/transformer_1d",
            "in_channels": 16,
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "cond_dim": 32,
        })
        
        x = torch.randn(1, 16, 32)  # [B, C, L]
        t = torch.tensor([500])
        
        output = backbone._forward_impl(x, t)
        assert output.shape == x.shape


class TestEquivariantGNNBackbone:
    def test_forward(self):
        from yggdrasil.blocks.backbones.equivariant_gnn import EquivariantGNNBackbone
        
        backbone = EquivariantGNNBackbone({
            "type": "backbone/equivariant_gnn",
            "hidden_dim": 64,
            "num_layers": 2,
            "in_features": 32,
            "coord_dim": 3,
        })
        
        # [B, N, features + 3 coords]
        x = torch.randn(1, 10, 35)
        t = torch.tensor([500])
        
        output = backbone._forward_impl(x, t)
        assert output.shape == x.shape
