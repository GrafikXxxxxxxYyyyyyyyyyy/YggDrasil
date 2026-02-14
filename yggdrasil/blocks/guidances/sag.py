"""Self-Attention Guidance (SAG) — чистый port-based блок.

Paper: "Improving Sample Quality of Diffusion Models Using Self-Attention Guidance"

SAG получает два выхода через порты:
- model_output (normal prediction)
- degraded_output (prediction from blurred input)

Формула: guided = model_output + scale * (model_output - degraded_output)

Граф должен выполнить blur -> backbone_degraded отдельным путём.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
from omegaconf import DictConfig

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import InputPort, OutputPort, TensorSpec
from yggdrasil.core.model.guidance import AbstractGuidance


@register_block("guidance/sag")
class SelfAttentionGuidance(AbstractGuidance):
    """Self-Attention Guidance.
    
    Порты:
        IN:  model_output (normal), degraded_output (from blurred input)
        OUT: guided_output
    """
    
    block_type = "guidance/sag"
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "guidance/sag"}
        super().__init__(config)
        self.scale = float(self.config.get("scale", 0.75))
    
    @classmethod
    def declare_io(cls) -> dict:
        return {
            "model_output": InputPort("model_output", spec=TensorSpec(space="latent"),
                                      description="Normal model prediction"),
            "degraded_output": InputPort("degraded_output", spec=TensorSpec(space="latent"),
                                         optional=True,
                                         description="Prediction from blurred/degraded input"),
            "guided_output": OutputPort("guided_output", spec=TensorSpec(space="latent"),
                                         description="Guided output"),
        }
    
    def process(self, **port_inputs) -> dict:
        model_output = port_inputs.get("model_output")
        degraded_output = port_inputs.get("degraded_output")
        
        if model_output is None or self.scale <= 0.0 or degraded_output is None:
            return {"guided_output": model_output, "output": model_output}
        
        guided = model_output + self.scale * (model_output - degraded_output)
        return {"guided_output": guided, "output": guided}
    
    def _forward_impl(self, *args, **kwargs) -> torch.Tensor:
        return args[0] if args else torch.zeros(1)
    
    def __call__(self, model_output, **kwargs) -> torch.Tensor:
        degraded = kwargs.get("degraded_output")
        if degraded is not None and self.scale > 0.0:
            return model_output + self.scale * (model_output - degraded)
        return model_output


@register_block("transform/gaussian_blur")
class GaussianBlurBlock(AbstractGuidance):
    """Gaussian blur — utility block for SAG pipelines.
    
    Применяет гауссово размытие к пространственным измерениям тензора.
    Используется как отдельный блок в графе перед backbone_degraded.
    """
    
    block_type = "transform/gaussian_blur"
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "transform/gaussian_blur"}
        super().__init__(config)
        self.sigma = float(self.config.get("sigma", 2.0))
    
    @classmethod
    def declare_io(cls) -> dict:
        return {
            "x": InputPort("x", spec=TensorSpec(space="latent"), description="Input tensor"),
            "blurred": OutputPort("blurred", spec=TensorSpec(space="latent"), description="Blurred output"),
            "guided_output": OutputPort("guided_output", spec=TensorSpec(space="latent"), description="Alias"),
        }
    
    def process(self, **kw) -> dict:
        x = kw.get("x") or kw.get("model_output")
        if x is None or x.dim() < 4:
            return {"blurred": x, "guided_output": x, "output": x}
        blurred = self._gaussian_blur(x)
        return {"blurred": blurred, "guided_output": blurred, "output": blurred}
    
    def _forward_impl(self, *args, **kwargs):
        return args[0] if args else torch.zeros(1)
    
    def __call__(self, model_output, **kwargs):
        return model_output
    
    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = int(self.sigma * 4) | 1
        kernel_size = max(3, kernel_size)
        coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
        gauss = torch.exp(-coords ** 2 / (2 * self.sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel = kernel.unsqueeze(0).unsqueeze(0).expand(x.shape[1], -1, -1, -1)
        padding = kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])
