# yggdrasil/blocks/adapters/t2i_adapter.py
from __future__ import annotations

import torch
from omegaconf import DictConfig
from typing import Optional, Dict, Any, List

from diffusers import T2IAdapter as DiffusersT2IAdapter
from diffusers.models import UNet2DConditionModel

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import InputPort, OutputPort
from yggdrasil.core.block.slot import Slot
from yggdrasil.core.block.base import AbstractBlock
from .base import AbstractAdapter
from yggdrasil.core.model.backbone import AbstractBackbone


@register_block("adapter/t2i")
class T2IAdapter(AbstractAdapter):
    """T2I-Adapter — лёгкий структурный контроллер. Может использоваться как узел графа (control_image -> output) или через inject_into."""
    
    block_type = "adapter/t2i"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.adapter_type = config.get("adapter_type", "depth")
        # Hugging Face repo uses underscores: TencentARC/t2iadapter_depth_sd15v2
        _default_pretrained = f"TencentARC/t2iadapter_{self.adapter_type}_sd15v2"
        self.pretrained = config.get("pretrained", _default_pretrained)
        self.scale = config.get("scale", 1.0)
        
        self.t2i_adapter = DiffusersT2IAdapter.from_pretrained(
            self.pretrained,
            torch_dtype=torch.float16 if config.get("fp16", True) else torch.float32,
        )
        self.t2i_adapter.requires_grad_(config.get("trainable", False))
        
        self.adapter_features: Optional[torch.Tensor] = None

    @classmethod
    def declare_io(cls):
        return {
            "control_image": InputPort("control_image", description="Conditioning image (depth, sketch, etc.)", optional=True),
            "sample": InputPort("sample", description="Noisy latent sample", optional=True),
            "timestep": InputPort("timestep", description="Timestep", optional=True),
            "encoder_hidden_states": InputPort("encoder_hidden_states", description="Text embeddings", optional=True),
            "output": OutputPort("output", description="Dict for backbone adapter_features (down_block_additional_residuals, mid_block_additional_residual)"),
        }
    
    def process(self, **kw) -> Dict[str, Any]:
        """Node mode: take control_image and return adapter_features for backbone. When control_image is None, return zero residuals."""
        control_image = kw.get("control_image")
        if control_image is None or self.t2i_adapter is None:
            return {"output": {"down_block_additional_residuals": None, "mid_block_additional_residual": None}}
        target_dtype = next(self.t2i_adapter.parameters()).dtype
        target_device = next(self.t2i_adapter.parameters()).device
        if control_image.dtype != target_dtype or control_image.device != target_device:
            control_image = control_image.to(device=target_device, dtype=target_dtype)
        if control_image.max() > 1.0:
            control_image = (control_image + 1.0) / 2.0
        with torch.no_grad():
            out = self.t2i_adapter(control_image)
        if isinstance(out, (tuple, list)):
            down_res = [x * self.scale for x in out]
            mid_res = None
        else:
            down_res = out if isinstance(out, list) else [out]
            down_res = [x * self.scale for x in down_res]
            mid_res = None
        return {"output": {"down_block_additional_residuals": down_res, "mid_block_additional_residual": mid_res}}
    
    def _define_slots(self):
        return {
            "conditioner": Slot(
                name="conditioner",
                accepts=AbstractBlock,
                multiple=True,
                optional=True
            )
        }
    
    def inject_into(self, target: AbstractBackbone):
        if hasattr(target, "unet") and isinstance(target.unet, UNet2DConditionModel):
            original_forward = target.unet.forward
            
            def t2i_forward(*args, **kwargs):
                if self.adapter_features is not None:
                    kwargs["adapter_features"] = self.adapter_features
                return original_forward(*args, **kwargs)
            
            target.unet.forward = t2i_forward
        else:
            target.add_pre_hook(self._pre_forward_hook)
    
    def _pre_forward_hook(self, module, x, timestep, condition: Dict[str, Any], **kwargs):
        """Хук перед forward backbone."""
        control_image = condition.get("control_image")
        if control_image is None:
            return x, timestep, condition
        
        if control_image.max() > 1.0:
            control_image = (control_image + 1.0) / 2.0
        
        with torch.no_grad():
            self.adapter_features = self.t2i_adapter(control_image)
        
        condition["adapter_features"] = self.adapter_features * self.scale
        return x, timestep, condition
    
    def apply(self, output: torch.Tensor, context=None) -> torch.Tensor:
        return output
    
    def __repr__(self):
        return f"<T2IAdapter type={self.adapter_type} scale={self.scale}>"