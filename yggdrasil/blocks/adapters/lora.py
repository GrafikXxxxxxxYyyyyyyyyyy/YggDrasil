"""Universal LoRA — apply to ANY block in the graph.

Key principle: LoRA is a Lego piece that can be applied to any nn.Module-based block.

Usage:

    # Apply to backbone
    lora = LoRAAdapter({"rank": 16, "target_modules": ["to_q", "to_k", "to_v"]})
    lora.inject_into(graph.nodes["backbone"])
    
    # Apply to text encoder  
    lora_te = LoRAAdapter({"rank": 8, "target_modules": ["q_proj", "v_proj"]})
    lora_te.inject_into(graph.nodes["text_encoder"])
    
    # Graph-level utility: apply LoRA to multiple nodes at once
    apply_lora(graph, {
        "backbone": {"rank": 16, "alpha": 16, "target_modules": ["to_q", "to_k", "to_v"]},
        "text_encoder": {"rank": 8, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    })
    
    # Train only LoRA weights
    trainer = GraphTrainer(graph, train_nodes=["backbone", "text_encoder"])
    # All base weights are frozen, only LoRA weights are trainable
    
    # Save/load LoRA weights separately
    save_lora(graph, "my_lora_weights/")
    load_lora(graph, "my_lora_weights/")
    
    # Merge LoRA into base weights (for deployment)
    merge_lora(graph, node_names=["backbone"])
"""
from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path
from omegaconf import DictConfig

from .base import AbstractAdapter
from ...core.block.base import AbstractBlock
from ...core.block.registry import register_block

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """A single LoRA layer that wraps a Linear or Conv2d module.
    
    Implements: output = base_output + dropout(x @ B^T @ A^T) * scaling
    """
    
    def __init__(
        self, 
        base_module: nn.Module, 
        rank: int = 16, 
        alpha: float = 16.0, 
        dropout: float = 0.0,
        init_strategy: str = "kaiming",  # "kaiming", "gaussian", "zeros"
    ):
        super().__init__()
        self.base_module = base_module
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merged = False
        
        if isinstance(base_module, nn.Linear):
            in_features = base_module.in_features
            out_features = base_module.out_features
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self._is_conv = False
        elif isinstance(base_module, nn.Conv2d):
            in_channels = base_module.in_channels
            out_channels = base_module.out_channels
            kernel_size = base_module.kernel_size
            self.lora_A = nn.Parameter(
                torch.zeros(rank, in_channels, *kernel_size)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(out_channels, rank, 1, 1)
            )
            self._is_conv = True
        else:
            raise TypeError(f"LoRA unsupported for {type(base_module)}")
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        self._init_weights(init_strategy)
        
        # Freeze base weights
        base_module.weight.requires_grad = False
        if base_module.bias is not None:
            base_module.bias.requires_grad = False
    
    def _init_weights(self, strategy: str):
        if strategy == "kaiming":
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
        elif strategy == "gaussian":
            nn.init.normal_(self.lora_A, std=0.01)
            nn.init.zeros_(self.lora_B)
        else:  # zeros
            nn.init.zeros_(self.lora_A)
            nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_module._original_forward(x)
        
        if self.merged:
            return base_out
        
        # Compute LoRA delta
        if self._is_conv:
            lora_out = F.conv2d(
                self.dropout(x), self.lora_A,
                stride=self.base_module.stride,
                padding=self.base_module.padding,
                dilation=self.base_module.dilation,
                groups=self.base_module.groups,
            )
            lora_out = F.conv2d(lora_out, self.lora_B)
        else:
            lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        
        return base_out + lora_out * self.scaling
    
    def merge(self):
        """Merge LoRA weights into base weights (permanent)."""
        if self.merged:
            return
        with torch.no_grad():
            if self._is_conv:
                # For conv: weight += B @ A * scaling
                delta = (self.lora_B @ self.lora_A.view(self.rank, -1)).view_as(
                    self.base_module.weight
                )
            else:
                delta = self.lora_B @ self.lora_A
            self.base_module.weight.data += delta * self.scaling
        self.merged = True
    
    def unmerge(self):
        """Unmerge LoRA weights from base weights."""
        if not self.merged:
            return
        with torch.no_grad():
            if self._is_conv:
                delta = (self.lora_B @ self.lora_A.view(self.rank, -1)).view_as(
                    self.base_module.weight
                )
            else:
                delta = self.lora_B @ self.lora_A
            self.base_module.weight.data -= delta * self.scaling
        self.merged = False


@register_block("adapter/lora")
class LoRAAdapter(AbstractAdapter):
    """Universal LoRA — works with ANY nn.Module-based block.
    
    Config:
        rank: int = 16  — LoRA rank
        alpha: float = 16.0  — scaling factor
        dropout: float = 0.0 
        target_modules: list[str] — module name patterns to apply LoRA to
        init_strategy: str = "kaiming" — weight init ("kaiming", "gaussian", "zeros")
    """
    
    block_type = "adapter/lora"
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "adapter/lora"}
        if isinstance(config, dict) and "type" not in config:
            config["type"] = "adapter/lora"
        super().__init__(config if isinstance(config, DictConfig) else DictConfig(config))
        
        self.rank = int(self.config.get("rank", 16))
        self.alpha = float(self.config.get("alpha", 16.0))
        self.dropout = float(self.config.get("dropout", 0.0))
        self.target_modules = list(self.config.get(
            "target_modules", ["to_q", "to_k", "to_v", "to_out.0"]
        ))
        self.init_strategy = str(self.config.get("init_strategy", "kaiming"))
        
        self.scaling = self.alpha / self.rank
        self.lora_layers: Dict[str, LoRALayer] = {}
        self._injected_target: Optional[nn.Module] = None
    
    def inject_into(self, target: nn.Module) -> LoRAAdapter:
        """Inject LoRA layers into target module (any nn.Module-based block).
        
        Finds all Linear/Conv2d layers matching target_modules patterns
        and wraps them with LoRA.
        
        Args:
            target: Any nn.Module (AbstractBlock, AbstractBackbone, etc.)
            
        Returns:
            self for chaining
        """
        self._injected_target = target
        count = 0
        
        for name, module in target.named_modules():
            if self._should_apply(name, module):
                lora = LoRALayer(
                    module, self.rank, self.alpha, self.dropout, self.init_strategy
                )
                self.lora_layers[name] = lora
                
                # Save original forward and replace
                module._original_forward = module.forward
                module.forward = lora.forward
                
                count += 1
        
        total_lora_params = sum(
            l.lora_A.numel() + l.lora_B.numel() for l in self.lora_layers.values()
        )
        logger.info(
            f"LoRA injected into {count} layers | "
            f"Rank: {self.rank} | Alpha: {self.alpha} | "
            f"LoRA params: {total_lora_params:,}"
        )
        
        return self
    
    def _should_apply(self, name: str, module: nn.Module) -> bool:
        """Check if LoRA should be applied to this module."""
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            return False
        return any(pattern in name for pattern in self.target_modules)
    
    def apply(self, output: torch.Tensor, context=None):
        """LoRA is already injected — passthrough."""
        return output
    
    def remove(self):
        """Remove LoRA from injected target, restoring original forwards."""
        for name, lora in self.lora_layers.items():
            lora.base_module.forward = lora.base_module._original_forward
            delattr(lora.base_module, '_original_forward')
            lora.base_module.weight.requires_grad = True
            if lora.base_module.bias is not None:
                lora.base_module.bias.requires_grad = True
        self.lora_layers.clear()
        self._injected_target = None
    
    def merge(self):
        """Merge all LoRA weights into base weights."""
        for lora in self.lora_layers.values():
            lora.merge()
        logger.info("LoRA merged into base weights")
    
    def unmerge(self):
        """Unmerge all LoRA weights from base weights."""
        for lora in self.lora_layers.values():
            lora.unmerge()
        logger.info("LoRA unmerged from base weights")
    
    def state_dict(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Return only LoRA weights."""
        state = {}
        for name, lora in self.lora_layers.items():
            state[f"{name}.lora_A"] = lora.lora_A.data
            state[f"{name}.lora_B"] = lora.lora_B.data
        return state
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], **kwargs):
        """Load LoRA weights."""
        for name, lora in self.lora_layers.items():
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in state_dict:
                lora.lora_A.data = state_dict[a_key].to(lora.lora_A.device)
            if b_key in state_dict:
                lora.lora_B.data = state_dict[b_key].to(lora.lora_B.device)
    
    def parameters(self, recurse=True):
        """Return only LoRA trainable parameters."""
        for lora in self.lora_layers.values():
            yield lora.lora_A
            yield lora.lora_B
    
    def named_parameters(self, prefix='', recurse=True):
        for name, lora in self.lora_layers.items():
            yield f"{prefix}{name}.lora_A", lora.lora_A
            yield f"{prefix}{name}.lora_B", lora.lora_B
    
    def num_parameters(self) -> int:
        return sum(l.lora_A.numel() + l.lora_B.numel() for l in self.lora_layers.values())


@register_block("adapter/dora")
class DoRAAdapter(LoRAAdapter):
    """DoRA (Weight-Decomposed Low-Rank Adaptation) — improved LoRA variant.
    
    Decomposes weight updates into magnitude and direction components.
    """
    block_type = "adapter/dora"
    
    def inject_into(self, target: nn.Module) -> DoRAAdapter:
        """Inject DoRA into target. Same as LoRA but with magnitude vectors."""
        super().inject_into(target)
        
        # Add magnitude vectors for DoRA
        self._magnitude_vectors = {}
        for name, lora in self.lora_layers.items():
            if not lora._is_conv:
                # Compute initial magnitude from base weight
                weight_norm = lora.base_module.weight.data.norm(dim=1, keepdim=True)
                self._magnitude_vectors[name] = nn.Parameter(weight_norm.squeeze())
        
        return self
    
    def parameters(self, recurse=True):
        """Return LoRA + magnitude parameters."""
        yield from super().parameters(recurse)
        for m in self._magnitude_vectors.values():
            yield m


# ==================== GRAPH-LEVEL UTILITIES ====================


def apply_lora(
    graph: "ComputeGraph",
    node_configs: Dict[str, Dict[str, Any]],
) -> Dict[str, LoRAAdapter]:
    """Apply LoRA to multiple nodes in a graph at once.
    
    Args:
        graph: ComputeGraph instance
        node_configs: {node_name: lora_config_dict}
        
    Returns:
        Dict of {node_name: LoRAAdapter} instances
        
    Example::
    
        adapters = apply_lora(graph, {
            "backbone": {"rank": 16, "target_modules": ["to_q", "to_k", "to_v"]},
            "text_encoder": {"rank": 8, "target_modules": ["q_proj", "v_proj"]},
        })
        # Now train only LoRA params
        trainer = GraphTrainer(graph, train_nodes=list(adapters.keys()))
    """
    adapters = {}
    for node_name, config in node_configs.items():
        if node_name not in graph.nodes:
            logger.warning(f"Node '{node_name}' not found in graph. Skipping.")
            continue
        
        block = graph.nodes[node_name]
        adapter = LoRAAdapter(config)
        adapter.inject_into(block)
        adapters[node_name] = adapter
        
        # Store adapter reference on the block for later retrieval
        if not hasattr(block, '_lora_adapters'):
            block._lora_adapters = []
        block._lora_adapters.append(adapter)
    
    return adapters


def save_lora(
    graph: "ComputeGraph",
    path: str,
    node_names: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Save LoRA weights from graph nodes.
    
    Args:
        graph: ComputeGraph with LoRA-injected nodes
        path: Directory to save weights
        node_names: Optional list of node names (default: all with LoRA)
        metadata: Optional metadata to save alongside weights
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    lora_state = {}
    
    for name, block in graph.nodes.items():
        if node_names and name not in node_names:
            continue
        if hasattr(block, '_lora_adapters'):
            for i, adapter in enumerate(block._lora_adapters):
                key = f"{name}" if i == 0 else f"{name}_{i}"
                lora_state[key] = {
                    "state_dict": adapter.state_dict(),
                    "config": {
                        "rank": adapter.rank,
                        "alpha": adapter.alpha,
                        "dropout": adapter.dropout,
                        "target_modules": adapter.target_modules,
                    },
                }
    
    save_data = {"lora_state": lora_state}
    if metadata:
        save_data["metadata"] = metadata
    
    torch.save(save_data, path / "lora_weights.pt")
    logger.info(f"LoRA weights saved to {path} ({len(lora_state)} nodes)")


def load_lora(
    graph: "ComputeGraph",
    path: str,
    strict: bool = True,
) -> Dict[str, LoRAAdapter]:
    """Load LoRA weights into graph nodes.
    
    If LoRA is not yet injected, injects it with saved configs.
    
    Args:
        graph: ComputeGraph to load LoRA into
        path: Directory with saved weights
        strict: Whether to fail on missing nodes
        
    Returns:
        Dict of {node_name: LoRAAdapter}
    """
    path = Path(path)
    data = torch.load(path / "lora_weights.pt", weights_only=False)
    lora_state = data["lora_state"]
    
    adapters = {}
    for key, info in lora_state.items():
        node_name = key.split("_")[0] if "_" in key else key
        if node_name not in graph.nodes:
            if strict:
                raise ValueError(f"Node '{node_name}' not found in graph")
            logger.warning(f"Node '{node_name}' not found, skipping")
            continue
        
        block = graph.nodes[node_name]
        
        # Inject LoRA if not already present
        if not hasattr(block, '_lora_adapters') or not block._lora_adapters:
            adapter = LoRAAdapter(info["config"])
            adapter.inject_into(block)
            if not hasattr(block, '_lora_adapters'):
                block._lora_adapters = []
            block._lora_adapters.append(adapter)
        else:
            adapter = block._lora_adapters[0]
        
        adapter.load_state_dict(info["state_dict"])
        adapters[node_name] = adapter
    
    logger.info(f"LoRA weights loaded from {path} ({len(adapters)} nodes)")
    return adapters


def merge_lora(graph: "ComputeGraph", node_names: Optional[List[str]] = None):
    """Merge LoRA weights into base weights for deployment.
    
    After merging, the model runs without LoRA overhead.
    
    Args:
        graph: ComputeGraph with LoRA-injected nodes
        node_names: Optional list of nodes (default: all with LoRA)
    """
    for name, block in graph.nodes.items():
        if node_names and name not in node_names:
            continue
        if hasattr(block, '_lora_adapters'):
            for adapter in block._lora_adapters:
                adapter.merge()
            logger.info(f"LoRA merged for node '{name}'")


def unmerge_lora(graph: "ComputeGraph", node_names: Optional[List[str]] = None):
    """Unmerge LoRA weights from base weights.
    
    Args:
        graph: ComputeGraph with merged LoRA
        node_names: Optional list of nodes (default: all with LoRA)
    """
    for name, block in graph.nodes.items():
        if node_names and name not in node_names:
            continue
        if hasattr(block, '_lora_adapters'):
            for adapter in block._lora_adapters:
                adapter.unmerge()
            logger.info(f"LoRA unmerged for node '{name}'")


def get_lora_info(graph: "ComputeGraph") -> Dict[str, Dict[str, Any]]:
    """Get information about LoRA adapters in the graph."""
    info = {}
    for name, block in graph.nodes.items():
        if hasattr(block, '_lora_adapters'):
            for adapter in block._lora_adapters:
                info[name] = {
                    "rank": adapter.rank,
                    "alpha": adapter.alpha,
                    "num_layers": len(adapter.lora_layers),
                    "num_parameters": adapter.num_parameters(),
                    "target_modules": adapter.target_modules,
                    "merged": any(l.merged for l in adapter.lora_layers.values()),
                }
    return info
