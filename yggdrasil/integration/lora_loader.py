# yggdrasil/integration/lora_loader.py
"""Load HuggingFace/diffusers-format LoRA into YggDrasil graph.

Supports:
- SD 1.5 / SDXL: UNet backbones (StableDiffusionXLLoraLoaderMixin).
- FLUX / SD3: Transformer backbones (FluxTransformer2DModel, SD3Transformer2DModel) when wrapped with ._model.

Requires: peft, safetensors (optional).
Example repos: https://huggingface.co/OnMoon/loras (SD), FLUX LoRA repos.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _find_unet_backbone(graph):
    """Return (block_name, unet) for first backbone with .unet, or (None, None)."""
    for name, block in graph._iter_all_blocks():
        if getattr(block, "block_type", "").startswith("backbone/") and hasattr(block, "unet"):
            return name, block.unet
    return None, None


def _find_transformer_backbone(graph):
    """Return (block_name, transformer_model) for first backbone with ._model that supports load_lora_weights."""
    for name, block in graph._iter_all_blocks():
        if not getattr(block, "block_type", "").startswith("backbone/"):
            continue
        model = getattr(block, "_model", None)
        if model is not None and callable(getattr(model, "load_lora_weights", None)):
            return name, model
    return None, None


def load_lora_weights(
    graph,
    pretrained_model_name_or_path: str,
    *,
    weight_name: Optional[str] = None,
    adapter_name: Optional[str] = None,
    **kwargs,
) -> List[str]:
    """Load LoRA weights into the graph's UNet (SD 1.5/SDXL) or transformer (FLUX/SD3).

    Args:
        graph: ComputeGraph (denoise_loop.backbone with .unet or ._model).
        pretrained_model_name_or_path: HF model id or local path.
        weight_name: Specific .safetensors file. If None, first found in repo.
        adapter_name: Adapter name for multi-LoRA. Default "default".
        **kwargs: Passed to diffusers lora_state_dict / load_lora_weights (cache_dir, token, revision, etc.).

    Returns:
        List of component names that received LoRA (e.g. ["unet"], ["transformer"]).

    Raises:
        ImportError: If peft is not installed.
        ValueError: If no compatible backbone found or LoRA format invalid.
    """
    try:
        from diffusers.utils import USE_PEFT_BACKEND
        if not USE_PEFT_BACKEND:
            raise ImportError("PEFT backend is required. Install: pip install peft")
    except Exception as e:
        raise ImportError("LoRA loading requires diffusers with PEFT. Install: pip install peft") from e

    adapter_name = adapter_name or "default"
    loaded: List[str] = []

    # 1) Try UNet backbones (SD 1.5, SDXL)
    _name, unet = _find_unet_backbone(graph)
    if unet is not None:
        from diffusers.loaders.lora_pipeline import StableDiffusionXLLoraLoaderMixin
        kwargs.setdefault("return_lora_metadata", True)
        kwargs["unet_config"] = getattr(unet, "config", None)
        state_dict, network_alphas, metadata = StableDiffusionXLLoraLoaderMixin.lora_state_dict(
            pretrained_model_name_or_path,
            weight_name=weight_name,
            **kwargs,
        )
        if not state_dict or not any("lora" in k for k in state_dict):
            raise ValueError("Invalid or empty LoRA checkpoint.")
        StableDiffusionXLLoraLoaderMixin.load_lora_into_unet(
            state_dict,
            network_alphas,
            unet=unet,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=None,
            low_cpu_mem_usage=kwargs.get("low_cpu_mem_usage", True),
            hotswap=False,
        )
        loaded.append("unet")
        logger.info("LoRA loaded from %s into unet", pretrained_model_name_or_path)
        return loaded

    # 2) Try transformer backbones (FLUX, SD3) — ._model with load_lora_weights
    _name, transformer = _find_transformer_backbone(graph)
    if transformer is not None:
        transformer.load_lora_weights(
            pretrained_model_name_or_path,
            weight_name=weight_name,
            adapter_name=adapter_name,
            **kwargs,
        )
        loaded.append("transformer")
        logger.info("LoRA loaded from %s into transformer", pretrained_model_name_or_path)
        return loaded

    raise ValueError(
        "No backbone with .unet (SD 1.5/SDXL) or ._model with load_lora_weights (FLUX/SD3) found in graph."
    )
