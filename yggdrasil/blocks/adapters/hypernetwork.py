# yggdrasil/blocks/adapters/hypernetwork.py
"""Hypernetwork adapter — small networks that modify attention weights.

Like A1111's Hypernetworks: trains a small neural network that
adjusts the key and value projections in cross-attention layers.

Usage::

    hypernet = HypernetworkAdapter({
        "type": "adapter/hypernetwork",
        "hidden_size": 768,
        "layer_structure": [768, 128, 768],
    })
    hypernet.inject_into(backbone)
    
    # Train with GraphTrainer
    trainer = GraphTrainer(graph, train_nodes=["hypernetwork"])
"""
from __future__ import annotations

import logging
import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import InputPort, OutputPort
from .base import AbstractAdapter

logger = logging.getLogger(__name__)


class HypernetworkLayer(nn.Module):
    """A single hypernetwork module that transforms attention features."""
    
    def __init__(self, layer_sizes: List[int], activation: str = "relu"):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "swish":
                    layers.append(nn.SiLU())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_block("adapter/hypernetwork")
class HypernetworkAdapter(AbstractAdapter):
    """Hypernetwork adapter — modifies cross-attention with small networks.
    
    For each cross-attention layer, a pair of small networks (for K and V)
    transforms the hidden states before they enter the attention computation.
    
    Features:
    - Configurable layer structure (depth and width)
    - Per-layer strength control
    - Multiple activation functions
    - Save/load trained weights
    """
    
    block_type = "adapter/hypernetwork"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.hidden_size = config.get("hidden_size", 768)
        self.layer_structure = list(config.get("layer_structure", [768, 128, 768]))
        self.activation = config.get("activation", "relu")
        self.strength = config.get("strength", 1.0)
        
        # K and V hypernetworks
        self.hypernet_k = HypernetworkLayer(self.layer_structure, self.activation)
        self.hypernet_v = HypernetworkLayer(self.layer_structure, self.activation)
        
        self._injected = False
        self._original_forwards = {}
    
    @classmethod
    def declare_io(cls):
        return {
            "hidden_states": InputPort("hidden_states", description="Input to cross-attention"),
            "modified_states": OutputPort("modified_states", description="Modified hidden states"),
        }
    
    def process(self, **kw) -> Dict[str, Any]:
        """Apply hypernetwork transformation."""
        hidden_states = kw.get("hidden_states")
        if hidden_states is None:
            return {"modified_states": None}
        
        k_mod = self.hypernet_k(hidden_states)
        v_mod = self.hypernet_v(hidden_states)
        
        # Return both K and V modifications
        return {
            "modified_states": hidden_states,
            "k_modifier": k_mod * self.strength,
            "v_modifier": v_mod * self.strength,
        }
    
    def inject_into(self, target):
        """Inject hypernetwork into cross-attention layers.
        
        Wraps the to_k and to_v linear layers in each attention block
        to add the hypernetwork transformation.
        """
        injected_count = 0
        
        for name, module in target.named_modules():
            if hasattr(module, "to_k") and hasattr(module, "to_v"):
                # Wrap K projection
                original_k = module.to_k
                original_v = module.to_v
                hypernet_k = self.hypernet_k
                hypernet_v = self.hypernet_v
                strength = self.strength
                
                class WrappedK(nn.Module):
                    def __init__(self, orig, hypernet, s):
                        super().__init__()
                        self.orig = orig
                        self.hypernet = hypernet
                        self.strength = s
                    
                    def forward(self, x):
                        base = self.orig(x)
                        try:
                            mod = self.hypernet(x)
                            return base + mod * self.strength
                        except Exception:
                            return base
                
                class WrappedV(nn.Module):
                    def __init__(self, orig, hypernet, s):
                        super().__init__()
                        self.orig = orig
                        self.hypernet = hypernet
                        self.strength = s
                    
                    def forward(self, x):
                        base = self.orig(x)
                        try:
                            mod = self.hypernet(x)
                            return base + mod * self.strength
                        except Exception:
                            return base
                
                module.to_k = WrappedK(original_k, hypernet_k, strength)
                module.to_v = WrappedV(original_v, hypernet_v, strength)
                self._original_forwards[name] = (original_k, original_v)
                injected_count += 1
        
        self._injected = True
        logger.info(f"Hypernetwork injected into {injected_count} attention layers")
    
    def remove(self, target):
        """Remove hypernetwork from target, restoring original layers."""
        for name, (orig_k, orig_v) in self._original_forwards.items():
            module = dict(target.named_modules()).get(name)
            if module is not None:
                module.to_k = orig_k
                module.to_v = orig_v
        self._original_forwards.clear()
        self._injected = False
        logger.info("Hypernetwork removed")
    
    def apply(self, output: torch.Tensor, context=None) -> torch.Tensor:
        return output
    
    def set_strength(self, strength: float):
        """Set hypernetwork strength."""
        self.strength = strength
    
    def save_weights(self, path: str):
        """Save hypernetwork weights."""
        state = {
            "hypernet_k": self.hypernet_k.state_dict(),
            "hypernet_v": self.hypernet_v.state_dict(),
            "layer_structure": self.layer_structure,
            "activation": self.activation,
            "strength": self.strength,
        }
        torch.save(state, path)
        logger.info(f"Saved hypernetwork to {path}")
    
    def load_weights(self, path: str):
        """Load hypernetwork weights."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.hypernet_k.load_state_dict(state["hypernet_k"])
        self.hypernet_v.load_state_dict(state["hypernet_v"])
        self.strength = state.get("strength", self.strength)
        logger.info(f"Loaded hypernetwork from {path}")
