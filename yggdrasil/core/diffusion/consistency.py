import torch
from yggdrasil.core.block.base import AbstractBlock
from yggdrasil.core.block.registry import register_block
from .process import AbstractDiffusionProcess


@register_block("diffusion/process/consistency")
class ConsistencyProcess(AbstractDiffusionProcess):
    """Consistency Models / Consistency Distillation."""
    block_type = "diffusion/process/consistency"
    
    def forward_process(self, x0, t, noise=None):
        # Упрощённо
        return {"xt": x0, "target": x0}
    
    def reverse_step(self, model_output, xt, t, **kwargs):
        return model_output  # consistency напрямую предсказывает x0
    
    def predict_x0(self, model_output, xt, t):
        return model_output