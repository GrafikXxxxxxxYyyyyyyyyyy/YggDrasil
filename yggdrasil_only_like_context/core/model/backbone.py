from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Any, Dict
from omegaconf import DictConfig

from ...core.block.base import AbstractBaseBlock
from ...core.block.registry import register_block
from ...core.block.port import Port, InputPort, OutputPort, TensorSpec


@register_block("backbone/abstract")
class AbstractBackbone(AbstractBaseBlock, nn.Module):
    """Абстрактный backbone (UNet, DiT, Transformer, GNN и т.д.).
    
    Контракт: реализовать ``process()`` или ``_forward_impl()``.
    
    Порты:
        IN:  x, timestep, condition (opt), position_embedding (opt), adapter_features (opt)
        OUT: output
    
    Пример::
    
        @register_block("backbone/my_unet")
        class MyUNet(AbstractBackbone):
            block_type = "backbone/my_unet"
            def __init__(self, config):
                super().__init__(config)
                self.model = ...
            def _forward_impl(self, x, timestep, condition=None, **kw):
                return self.model(x, timestep, condition)
    """
    
    block_type = "backbone/abstract"
    
    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        return {
            "x": InputPort("x", spec=TensorSpec(space="latent"), description="Latent input"),
            "timestep": InputPort("timestep", data_type="tensor", description="Timestep"),
            "condition": InputPort("condition", data_type="any", optional=True, description="Condition embeddings (dict or tensor; tensor is wrapped as encoder_hidden_states)"),
            "position_embedding": InputPort("position_embedding", optional=True, description="Position embedding"),
            "adapter_features": InputPort("adapter_features", data_type="any", optional=True, description="Adapter features (ControlNet, T2I) — dict or list of dicts"),
            "output": OutputPort("output", spec=TensorSpec(space="latent"), description="Denoised output"),
        }
    
    def process(self, **port_inputs) -> Dict[str, Any]:
        x = port_inputs.get("x")
        timestep = port_inputs.get("timestep")
        condition = port_inputs.get("condition")
        position_embedding = port_inputs.get("position_embedding")
        
        # Merge list of condition dicts (from graph: multiple conditioners -> one backbone)
        if isinstance(condition, list):
            merged = {}
            for c in condition:
                if isinstance(c, dict):
                    merged.update(c)
            condition = merged if merged else None
        # Normalize: bare tensor -> dict
        if condition is not None and not isinstance(condition, dict):
            condition = {"encoder_hidden_states": condition}
        
        extra = {k: v for k, v in port_inputs.items()
                 if k not in ("x", "timestep", "condition", "position_embedding")}
        output = self.forward(
            x=x, timestep=timestep, condition=condition,
            position_embedding=position_embedding, **extra,
        )
        return {"output": output}
    
    def _forward_impl(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        position_embedding: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """Override this or process() for custom backbone logic."""
        raise NotImplementedError(
            f"{type(self).__name__} должен реализовать process() или _forward_impl()"
        )
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        output = super().forward(*args, **kwargs)
        # Adapters are wired via graph (adapter node -> backbone.adapter_features); no slot
        return output
    
    def inject_adapter(self, adapter: AbstractBaseBlock):
        """No-op: adapters are wired via graph (adapter node -> backbone.adapter_features)."""
        pass