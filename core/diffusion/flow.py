import torch
from abc import abstractmethod
from omegaconf import DictConfig

from ...core.block.base import AbstractBlock
from ...core.block.registry import register_block
from .process import AbstractDiffusionProcess


@register_block("diffusion/process/flow/abstract")
class AbstractFlowProcess(AbstractDiffusionProcess):
    """Базовый flow-matching процесс."""
    block_type = "diffusion/process/flow/abstract"


@register_block("diffusion/process/flow/rectified")
class RectifiedFlowProcess(AbstractFlowProcess):
    """Rectified Flow (Flux-style)."""
    block_type = "diffusion/process/flow/rectified"
    
    def forward_process(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return {"xt": (1 - t) * x0 + t * noise, "target": noise - x0}
    
    def reverse_step(self, model_output, xt, t, **kwargs):
        # velocity → next
        return xt + model_output * (t[0] - t[1])
    
    def predict_x0(self, model_output, xt, t):
        return xt - t * model_output
    
    def predict_velocity(self, model_output, xt, t):
        return model_output