# yggdrasil/integration/lora_loader.py
"""Load HuggingFace/diffusers-format LoRA into YggDrasil graph (UNet / text encoder).

Requires: peft, safetensors (optional). Uses diffusers StableDiffusionXLLoraLoaderMixin.
Example repos: https://huggingface.co/OnMoon/loras
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def load_lora_weights(
    graph,
    pretrained_model_name_or_path: str,
    *,
    weight_name: Optional[str] = None,
    adapter_name: Optional[str] = None,
    **kwargs,
) -> List[str]:
    """Load LoRA weights from HuggingFace repo or path into the graph's UNet (and optionally text encoders).

    Uses diffusers' StableDiffusionXLLoraLoaderMixin; requires PEFT backend.

    Args:
        graph: ComputeGraph (with denoise_loop.backbone or similar containing .unet).
        pretrained_model_name_or_path: HF model id (e.g. "OnMoon/loras") or local path.
        weight_name: Specific .safetensors file (e.g. "MyLora.safetensors"). If None, first found in repo.
        adapter_name: Adapter name for multi-LoRA. Default "default".
        **kwargs: Passed to diffusers lora_state_dict (cache_dir, token, revision, etc.).

    Returns:
        List of component names that received LoRA (e.g. ["unet"]).

    Raises:
        ImportError: If peft is not installed.
        ValueError: If no backbone with .unet found or LoRA format invalid.
    """
    try:
        from diffusers.utils import USE_PEFT_BACKEND
        if not USE_PEFT_BACKEND:
            raise ImportError("PEFT backend is required. Install: pip install peft")
    except Exception as e:
        raise ImportError("LoRA loading requires diffusers with PEFT. Install: pip install peft") from e

    from diffusers.loaders.lora_pipeline import StableDiffusionXLLoraLoaderMixin

    # Find backbone with .unet (batched or non-batched)
    unet = None
    for _name, block in graph._iter_all_blocks():
        if getattr(block, "block_type", "").startswith("backbone/") and hasattr(block, "unet"):
            unet = block.unet
            break
    if unet is None:
        raise ValueError("No backbone with .unet found in graph (required for LoRA).")

    # Load state dict via SDXL mixin (class methods)
    kwargs.setdefault("return_lora_metadata", True)
    kwargs["unet_config"] = getattr(unet, "config", None)
    state_dict, network_alphas, metadata = StableDiffusionXLLoraLoaderMixin.lora_state_dict(
        pretrained_model_name_or_path,
        weight_name=weight_name,
        **kwargs,
    )
    if not state_dict or not any("lora" in k for k in state_dict):
        raise ValueError("Invalid or empty LoRA checkpoint.")

    adapter_name = adapter_name or "default"
    loaded = []

    # UNet
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

    logger.info("LoRA loaded from %s into %s", pretrained_model_name_or_path, loaded)
    return loaded
