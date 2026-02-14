"""Distributed training utilities for YggDrasil.

Supports:
- DDP (DistributedDataParallel) — basic multi-GPU
- FSDP (FullyShardedDataParallel) — memory-efficient for large models
- DeepSpeed — ZeRO stages 1-3, offloading

All strategies work with GraphTrainer — you just wrap the graph before training.

Usage:

    # DDP
    from yggdrasil.training.distributed import wrap_ddp, setup_distributed
    setup_distributed()
    graph = ComputeGraph.from_template("sd15_txt2img")
    graph = wrap_ddp(graph, train_nodes=["backbone"])
    trainer = GraphTrainer(graph, train_nodes=["backbone"])
    trainer.train(dataset)
    
    # FSDP  
    from yggdrasil.training.distributed import wrap_fsdp
    graph = wrap_fsdp(graph, train_nodes=["backbone"])
    
    # DeepSpeed
    from yggdrasil.training.distributed import DeepSpeedGraphTrainer
    trainer = DeepSpeedGraphTrainer(graph, train_nodes=["backbone"], ds_config={...})
"""
from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ==================== SETUP ====================

def setup_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
):
    """Initialize distributed process group.
    
    Call this at the start of each process.
    Typically called in the training script before creating graph/trainer.
    """
    if torch.distributed.is_initialized():
        return
    
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size <= 1:
        logger.info("World size <= 1, skipping distributed setup")
        return
    
    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    
    logger.info(f"Distributed initialized: rank={rank}, world_size={world_size}")


def cleanup_distributed():
    """Clean up distributed process group."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def get_rank() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


# ==================== DDP ====================

def wrap_ddp(
    graph: "ComputeGraph",
    train_nodes: List[str],
    device_ids: Optional[List[int]] = None,
    find_unused_parameters: bool = False,
) -> "ComputeGraph":
    """Wrap train_nodes in a graph with DistributedDataParallel.
    
    Only the specified train_nodes get DDP-wrapped.
    All other nodes remain unchanged.
    
    Args:
        graph: ComputeGraph instance
        train_nodes: List of node names to wrap with DDP
        device_ids: GPU device IDs (default: [LOCAL_RANK])
        find_unused_parameters: DDP option for models with unused params
        
    Returns:
        Modified graph (in-place)
    """
    if not torch.distributed.is_initialized():
        logger.warning("Distributed not initialized. Skipping DDP wrapping.")
        return graph
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device_ids is None and torch.cuda.is_available():
        device_ids = [local_rank]
    
    for name in train_nodes:
        if name not in graph.nodes:
            logger.warning(f"Node '{name}' not found in graph. Skipping DDP.")
            continue
        
        block = graph.nodes[name]
        if not isinstance(block, nn.Module):
            logger.warning(f"Node '{name}' is not nn.Module. Skipping DDP.")
            continue
        
        # Move to correct device
        if torch.cuda.is_available():
            block = block.to(f"cuda:{local_rank}")
        
        wrapped = nn.parallel.DistributedDataParallel(
            block,
            device_ids=device_ids,
            find_unused_parameters=find_unused_parameters,
        )
        graph.nodes[name] = wrapped
        logger.info(f"DDP wrapped: {name}")
    
    return graph


# ==================== FSDP ====================

def wrap_fsdp(
    graph: "ComputeGraph",
    train_nodes: List[str],
    sharding_strategy: str = "FULL_SHARD",
    cpu_offload: bool = False,
    mixed_precision: Optional[str] = None,
) -> "ComputeGraph":
    """Wrap train_nodes with FullyShardedDataParallel.
    
    FSDP shards model parameters across GPUs for memory efficiency.
    Useful for very large models (SDXL, FLUX, etc.)
    
    Args:
        graph: ComputeGraph instance
        train_nodes: Nodes to wrap
        sharding_strategy: "FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"
        cpu_offload: Whether to offload to CPU
        mixed_precision: "fp16" or "bf16" (None for full precision)
    """
    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            CPUOffload,
            MixedPrecision,
        )
    except ImportError:
        logger.error("FSDP not available. Requires PyTorch >= 1.12")
        return graph
    
    if not torch.distributed.is_initialized():
        logger.warning("Distributed not initialized. Skipping FSDP.")
        return graph
    
    # Resolve sharding strategy
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    strategy = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)
    
    # CPU offload
    offload = CPUOffload(offload_params=True) if cpu_offload else None
    
    # Mixed precision
    mp = None
    if mixed_precision == "fp16":
        mp = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    elif mixed_precision == "bf16":
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    
    for name in train_nodes:
        if name not in graph.nodes:
            continue
        
        block = graph.nodes[name]
        if not isinstance(block, nn.Module):
            continue
        
        wrapped = FSDP(
            block,
            sharding_strategy=strategy,
            cpu_offload=offload,
            mixed_precision=mp,
        )
        graph.nodes[name] = wrapped
        logger.info(f"FSDP wrapped: {name} (strategy={sharding_strategy})")
    
    return graph


# ==================== DEEPSPEED ====================

class DeepSpeedGraphTrainer:
    """GraphTrainer with DeepSpeed integration.
    
    Supports ZeRO stages 1-3 with optional CPU/NVMe offloading.
    
    Usage::
    
        ds_config = {
            "train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "optimizer": {"type": "AdamW", "params": {"lr": 1e-4}},
            "fp16": {"enabled": True},
            "zero_optimization": {"stage": 2},
        }
        
        trainer = DeepSpeedGraphTrainer(
            graph=graph,
            train_nodes=["backbone"],
            ds_config=ds_config,
        )
        trainer.train(dataset)
    """
    
    def __init__(
        self,
        graph: "ComputeGraph",
        train_nodes: List[str],
        ds_config: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ):
        self.graph = graph
        self.train_nodes = train_nodes
        self.ds_config = ds_config
        
        from .graph_trainer import GraphTrainingConfig
        self.config = GraphTrainingConfig.from_dict(config or {})
        
        self._engine = None
        self._step_count = 0
    
    def setup(self):
        """Initialize DeepSpeed engine."""
        try:
            import deepspeed
        except ImportError:
            raise ImportError(
                "DeepSpeed not installed. Install with: pip install deepspeed"
            )
        
        # Collect trainable parameters from specified nodes
        model_params = []
        for name in self.train_nodes:
            block = self.graph.nodes.get(name)
            if block is not None and hasattr(block, 'parameters'):
                model_params.extend(block.parameters())
        
        # Create a wrapper module for DeepSpeed
        wrapper = _TrainableNodesWrapper(self.graph, self.train_nodes)
        
        self._engine, self._optimizer, _, self._scheduler = deepspeed.initialize(
            model=wrapper,
            config=self.ds_config,
        )
        
        logger.info(f"DeepSpeed initialized with ZeRO stage {self.ds_config.get('zero_optimization', {}).get('stage', 0)}")
        return self
    
    def train(self, dataset, **kwargs) -> Dict[str, Any]:
        """Run training with DeepSpeed."""
        self.setup()
        
        from .graph_trainer import GraphTrainingConfig
        from yggdrasil.core.graph.executor import GraphExecutor
        
        executor = GraphExecutor(no_grad=False)
        loader = dataset.get_dataloader(
            batch_size=self.ds_config.get("train_micro_batch_size_per_gpu", 1),
            shuffle=True,
        )
        
        history = {"loss": [], "step": []}
        
        for epoch in range(self.config.num_epochs):
            for batch in loader:
                # Prepare inputs
                inputs = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self._engine.device)
                    else:
                        inputs[k] = v
                
                # Forward
                outputs = executor.execute_training(self.graph, **inputs)
                
                # Loss
                prediction = outputs.get("noise_pred", outputs.get("output"))
                target = batch.get("target", batch.get("noise"))
                if isinstance(target, torch.Tensor):
                    target = target.to(prediction.device)
                loss = torch.nn.functional.mse_loss(prediction, target)
                
                # Backward + step (DeepSpeed handles gradient accumulation)
                self._engine.backward(loss)
                self._engine.step()
                
                self._step_count += 1
                history["loss"].append(loss.item())
                history["step"].append(self._step_count)
        
        return history


class _TrainableNodesWrapper(nn.Module):
    """Wrapper that makes trainable graph nodes visible to DeepSpeed."""
    
    def __init__(self, graph: "ComputeGraph", train_nodes: List[str]):
        super().__init__()
        self._modules_dict = nn.ModuleDict()
        for name in train_nodes:
            block = graph.nodes.get(name)
            if isinstance(block, nn.Module):
                self._modules_dict[name] = block
    
    def forward(self, *args, **kwargs):
        # Not used directly — graph executor handles the forward
        pass


# ==================== DATA UTILITIES ====================

def make_distributed_sampler(dataset, shuffle: bool = True):
    """Create a DistributedSampler for a dataset."""
    if not torch.distributed.is_initialized():
        return None
    return torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=shuffle,
        num_replicas=get_world_size(),
        rank=get_rank(),
    )


def make_distributed_dataloader(
    dataset,
    batch_size: int = 1,
    num_workers: int = 0,
    **kwargs,
):
    """Create a DataLoader with distributed sampling."""
    sampler = make_distributed_sampler(dataset, shuffle=kwargs.pop("shuffle", True))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        **kwargs,
    )
