"""LossBlock — loss functions as graph nodes.

Training graph pattern:
    [DatasetBlock] --> [backbone] --> [LossBlock] --> loss_value
                          |                ^
                    [conditioner]     [target_from_dataset]

LossBlock receives prediction and target through ports and outputs the loss.
This makes loss computation part of the graph — fully composable.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional
from omegaconf import DictConfig

from ...core.block.base import AbstractBlock
from ...core.block.registry import register_block
from ...core.block.port import InputPort, OutputPort, TensorSpec, Port


@register_block("loss/mse")
class MSELossBlock(AbstractBlock):
    """MSE Loss as a graph node."""
    
    block_type = "loss/mse"
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "loss/mse"}
        super().__init__(config)
        self.weight = float(self.config.get("weight", 1.0))
    
    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        return {
            "prediction": InputPort("prediction", description="Model prediction"),
            "target": InputPort("target", description="Ground truth target"),
            "weights": InputPort("weights", data_type="tensor", optional=True,
                                 description="Per-sample or per-timestep weights"),
            "loss": OutputPort("loss", data_type="tensor", description="Scalar loss value"),
        }
    
    def process(self, **kw) -> dict:
        pred = kw["prediction"]
        target = kw["target"]
        if isinstance(target, torch.Tensor):
            target = target.to(pred.device)
        
        loss = F.mse_loss(pred, target, reduction="none")
        
        weights = kw.get("weights")
        if weights is not None:
            loss = loss * weights.view(-1, *([1] * (loss.ndim - 1)))
        
        scalar_loss = loss.mean() * self.weight
        return {"loss": scalar_loss, "output": scalar_loss}
    
    def _forward_impl(self, **kwargs):
        return self.process(**kwargs)


@register_block("loss/l1")
class L1LossBlock(AbstractBlock):
    """L1 Loss as a graph node."""
    
    block_type = "loss/l1"
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "loss/l1"}
        super().__init__(config)
        self.weight = float(self.config.get("weight", 1.0))
    
    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        return {
            "prediction": InputPort("prediction"),
            "target": InputPort("target"),
            "loss": OutputPort("loss", data_type="tensor"),
        }
    
    def process(self, **kw) -> dict:
        loss = F.l1_loss(kw["prediction"], kw["target"].to(kw["prediction"].device))
        return {"loss": loss * self.weight, "output": loss * self.weight}
    
    def _forward_impl(self, **kwargs):
        return self.process(**kwargs)


@register_block("loss/composite")
class CompositeLossBlock(AbstractBlock):
    """Combine multiple loss blocks with weights.
    
    Config:
        components: list of {type, weight} dicts
    """
    
    block_type = "loss/composite"
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "loss/composite"}
        super().__init__(config)
        self._components = []
    
    def add_component(self, loss_block: AbstractBlock, weight: float = 1.0):
        """Add a loss component."""
        self._components.append((loss_block, weight))
        return self
    
    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        return {
            "prediction": InputPort("prediction"),
            "target": InputPort("target"),
            "loss": OutputPort("loss", data_type="tensor"),
        }
    
    def process(self, **kw) -> dict:
        total = torch.tensor(0.0, device=kw["prediction"].device if isinstance(kw.get("prediction"), torch.Tensor) else "cpu")
        for comp, w in self._components:
            result = comp.process(**kw)
            total = total + w * result["loss"]
        return {"loss": total, "output": total}
    
    def _forward_impl(self, **kwargs):
        return self.process(**kwargs)
