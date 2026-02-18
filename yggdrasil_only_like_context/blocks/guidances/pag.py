"""Perturbed Attention Guidance (PAG).

Improves generation quality by comparing normal attention
output with perturbed (identity) attention output.
"""
from __future__ import annotations

import torch
from typing import Optional, Dict, Any
from omegaconf import DictConfig

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.model.guidance import AbstractGuidance


@register_block("guidance/pag")
class PerturbedAttentionGuidance(AbstractGuidance):
    """Perturbed Attention Guidance (PAG).
    
    Works by replacing self-attention maps with identity matrices
    and using the difference as guidance signal. This improves
    structural coherence without requiring negative prompts.
    
    Paper: "Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance"
    """
    
    block_type = "guidance/pag"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.scale = float(config.get("scale", 3.0))
        self.layers = config.get("layers", "mid")  # which layers to perturb
    
    def _forward_impl(self, *args, **kwargs) -> torch.Tensor:
        return args[0] if args else torch.zeros(1)
    
    def __call__(
        self,
        model_output: torch.Tensor,
        condition: Optional[Dict[str, Any]] = None,
        model=None,
        x: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Apply PAG.
        
        For full implementation, hooks into the backbone's attention layers
        to replace attention maps with identity. Here we provide the
        structural framework that can be extended.
        """
        if self.scale <= 0.0 or model is None or x is None or t is None:
            return model_output
        
        nodes = getattr(getattr(model, "_graph", None), "nodes", None) or getattr(model, "_slot_children", {})
        guidance_keys = [k for k in (nodes or {}) if isinstance(k, str) and k.startswith("guidance_")]
        saved = {}
        if hasattr(model, "_graph") and model._graph and guidance_keys:
            for k in guidance_keys:
                saved[k] = model._graph.nodes.pop(k, None)
        try:
            hooks = self._install_perturbation_hooks(model)
            perturbed_output = model._forward_impl(x=x, t=t, condition=condition, return_dict=False)
            for h in hooks:
                h.remove()
            guided = model_output + self.scale * (model_output - perturbed_output)
            return guided
        finally:
            if hasattr(model, "_graph") and model._graph and saved:
                for k, v in saved.items():
                    if v is not None:
                        model._graph.nodes[k] = v
    
    def _install_perturbation_hooks(self, model) -> list:
        """Install hooks that replace attention with identity.
        
        Returns list of hook handles for cleanup.
        """
        hooks = []
        nodes = getattr(getattr(model, "_graph", None), "nodes", None) or getattr(model, "_slot_children", {})
        backbone = nodes.get("backbone") if nodes else None
        if backbone is None:
            return hooks
        
        # Find attention modules and replace with identity
        for name, module in backbone.named_modules():
            if "attn" in name.lower() and hasattr(module, "forward"):
                # Create a hook that replaces attention output with identity-like behavior
                def make_hook(mod):
                    def hook_fn(module, input, output):
                        # Identity attention: each token attends only to itself
                        if isinstance(output, tuple):
                            return output  # Don't perturb cross-attention
                        return output * 0.0 + input[0] if len(input) > 0 else output
                    return hook_fn
                
                h = module.register_forward_hook(make_hook(module))
                hooks.append(h)
        
        return hooks
