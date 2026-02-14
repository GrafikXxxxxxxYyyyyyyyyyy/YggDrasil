# yggdrasil/training/checkpoint_ops.py
"""Checkpoint operations — merge, extract, prune model weights.

Like A1111's checkpoint merger: combine models, extract LoRA-like deltas,
and prune unnecessary weights.

Usage::

    from yggdrasil.training.checkpoint_ops import (
        merge_checkpoints, extract_diff, prune_model
    )
    
    # Merge two models (weighted average)
    merged_graph = merge_checkpoints(graph_a, graph_b, alpha=0.5)
    
    # Extract LoRA-like delta
    delta = extract_diff(base_graph, finetuned_graph)
    
    # Prune small weights
    prune_model(graph, threshold=1e-4)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def merge_checkpoints(
    graph_a,
    graph_b,
    alpha: float = 0.5,
    nodes: Optional[List[str]] = None,
    method: str = "weighted_sum",
) -> None:
    """Merge weights of two graphs (in-place on graph_a).
    
    Args:
        graph_a: Base graph (modified in place)
        graph_b: Second graph to merge from
        alpha: Interpolation weight. 0.0 = keep graph_a, 1.0 = use graph_b
        nodes: Only merge specific nodes (None = all common nodes)
        method: "weighted_sum" (default), "add_difference", "sigmoid"
    
    Example::
    
        # 50/50 merge
        merge_checkpoints(model_a, model_b, alpha=0.5)
        
        # Keep 70% of model_a, 30% of model_b
        merge_checkpoints(model_a, model_b, alpha=0.3)
    """
    target_nodes = nodes or list(set(graph_a.nodes.keys()) & set(graph_b.nodes.keys()))
    
    merged_params = 0
    for node_name in target_nodes:
        block_a = graph_a.nodes.get(node_name)
        block_b = graph_b.nodes.get(node_name)
        
        if block_a is None or block_b is None:
            logger.warning(f"Merge: node '{node_name}' missing from one of the graphs, skipping")
            continue
        
        if not hasattr(block_a, 'state_dict') or not hasattr(block_b, 'state_dict'):
            continue
        
        sd_a = block_a.state_dict()
        sd_b = block_b.state_dict()
        
        merged_sd = {}
        for key in sd_a:
            if key in sd_b and sd_a[key].shape == sd_b[key].shape:
                if method == "weighted_sum":
                    merged_sd[key] = (1 - alpha) * sd_a[key] + alpha * sd_b[key]
                elif method == "add_difference":
                    # A + alpha * (B - A) which is same as weighted_sum
                    merged_sd[key] = sd_a[key] + alpha * (sd_b[key] - sd_a[key])
                elif method == "sigmoid":
                    # Sigmoid-weighted interpolation
                    t = torch.sigmoid(torch.tensor(alpha * 10 - 5))
                    merged_sd[key] = (1 - t) * sd_a[key] + t * sd_b[key]
                else:
                    merged_sd[key] = (1 - alpha) * sd_a[key] + alpha * sd_b[key]
                merged_params += sd_a[key].numel()
            else:
                merged_sd[key] = sd_a[key]
        
        block_a.load_state_dict(merged_sd, strict=False)
    
    logger.info(
        f"Merged {len(target_nodes)} nodes ({merged_params:,} params) "
        f"with alpha={alpha} method={method}"
    )


def extract_diff(
    base_graph,
    finetuned_graph,
    nodes: Optional[List[str]] = None,
    threshold: float = 0.0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Extract weight difference between base and fine-tuned models.
    
    Useful for creating LoRA-like adapters from two checkpoints,
    or for analyzing what changed during fine-tuning.
    
    Args:
        base_graph: Original/base model
        finetuned_graph: Fine-tuned model
        nodes: Only extract specific nodes (None = all common nodes)
        threshold: Minimum L2 norm for a parameter diff to be included
        
    Returns:
        Dict of {node_name: {param_name: delta_tensor}}
    
    Example::
    
        delta = extract_diff(base_model, my_finetune)
        # delta["backbone"]["linear.weight"] contains the weight difference
    """
    target_nodes = nodes or list(
        set(base_graph.nodes.keys()) & set(finetuned_graph.nodes.keys())
    )
    
    result = {}
    total_diffs = 0
    
    for node_name in target_nodes:
        base_block = base_graph.nodes.get(node_name)
        ft_block = finetuned_graph.nodes.get(node_name)
        
        if base_block is None or ft_block is None:
            continue
        if not hasattr(base_block, 'state_dict') or not hasattr(ft_block, 'state_dict'):
            continue
        
        sd_base = base_block.state_dict()
        sd_ft = ft_block.state_dict()
        
        diffs = {}
        for key in sd_base:
            if key in sd_ft and sd_base[key].shape == sd_ft[key].shape:
                delta = sd_ft[key] - sd_base[key]
                if threshold <= 0 or delta.norm().item() > threshold:
                    diffs[key] = delta
                    total_diffs += 1
        
        if diffs:
            result[node_name] = diffs
    
    logger.info(f"Extracted {total_diffs} parameter diffs from {len(result)} nodes")
    return result


def apply_diff(
    graph,
    diff: Dict[str, Dict[str, torch.Tensor]],
    scale: float = 1.0,
) -> None:
    """Apply a weight diff (from extract_diff) to a graph.
    
    Args:
        graph: Target graph to modify
        diff: Weight differences from extract_diff
        scale: Scale factor for the diff (0.5 = half strength)
    """
    for node_name, param_diffs in diff.items():
        block = graph.nodes.get(node_name)
        if block is None or not hasattr(block, 'state_dict'):
            continue
        
        sd = block.state_dict()
        for key, delta in param_diffs.items():
            if key in sd and sd[key].shape == delta.shape:
                sd[key] = sd[key] + scale * delta
        
        block.load_state_dict(sd, strict=False)
    
    logger.info(f"Applied diff to {len(diff)} nodes with scale={scale}")


def prune_model(
    graph,
    threshold: float = 1e-4,
    nodes: Optional[List[str]] = None,
    method: str = "magnitude",
) -> Dict[str, int]:
    """Prune small weights from model nodes.
    
    Args:
        graph: Graph to prune
        threshold: Prune weights with absolute value below this
        nodes: Only prune specific nodes (None = all)
        method: "magnitude" (zero out small weights), "unstructured" (same)
        
    Returns:
        Dict of {node_name: number_of_pruned_params}
    """
    target_nodes = nodes or list(graph.nodes.keys())
    results = {}
    
    for node_name in target_nodes:
        block = graph.nodes.get(node_name)
        if block is None or not hasattr(block, 'parameters'):
            continue
        
        pruned = 0
        for param in block.parameters():
            mask = param.data.abs() < threshold
            param.data[mask] = 0.0
            pruned += mask.sum().item()
        
        if pruned > 0:
            results[node_name] = int(pruned)
    
    total = sum(results.values())
    logger.info(f"Pruned {total:,} parameters across {len(results)} nodes (threshold={threshold})")
    return results


def save_diff(diff: Dict[str, Dict[str, torch.Tensor]], path: str):
    """Save extracted diff to file."""
    torch.save(diff, path)
    logger.info(f"Saved diff to {path}")


def load_diff(path: str) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load diff from file."""
    return torch.load(path, map_location="cpu", weights_only=True)
