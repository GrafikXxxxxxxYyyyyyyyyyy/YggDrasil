"""YggDrasil Hub — HuggingFace Hub integration.

Model registry + auto-download + caching for pretrained models.

    from yggdrasil.hub import resolve_model, download_model
    
    template, kwargs = resolve_model("stable-diffusion-v1-5/stable-diffusion-v1-5")
    # template = "sd15_txt2img"
    # kwargs = {"pretrained": "stable-diffusion-v1-5/stable-diffusion-v1-5"}
"""
from __future__ import annotations

import os
import logging
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Model Registry ──
# Maps HuggingFace model IDs to YggDrasil templates + configs.
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Stable Diffusion 1.x
    "stable-diffusion-v1-5/stable-diffusion-v1-5": {
        "template": "sd15_txt2img",
        "default_width": 512,
        "default_height": 512,
    },
    "runwayml/stable-diffusion-v1-5": {
        "template": "sd15_txt2img",
        "default_width": 512,
        "default_height": 512,
    },
    "CompVis/stable-diffusion-v1-4": {
        "template": "sd15_txt2img",
        "default_width": 512,
        "default_height": 512,
    },
    
    # SDXL
    "stabilityai/stable-diffusion-xl-base-1.0": {
        "template": "sdxl_txt2img",
        "default_width": 1024,
        "default_height": 1024,
    },
    "stabilityai/stable-diffusion-xl-refiner-1.0": {
        "template": "sdxl_refiner",
        "default_width": 1024,
        "default_height": 1024,
    },
    
    # SD 3
    "stabilityai/stable-diffusion-3-medium-diffusers": {
        "template": "sd3_txt2img",
        "default_width": 1024,
        "default_height": 1024,
    },
    
    # FLUX.1
    "black-forest-labs/FLUX.1-dev": {
        "template": "flux_txt2img",
        "default_width": 1024,
        "default_height": 1024,
    },
    "black-forest-labs/FLUX.1-schnell": {
        "template": "flux_schnell_txt2img",
        "default_width": 1024,
        "default_height": 1024,
    },
    
    # FLUX.2
    "black-forest-labs/FLUX.2-dev": {
        "template": "flux2_txt2img",
        "default_width": 1024,
        "default_height": 1024,
    },
    "black-forest-labs/FLUX.2-schnell": {
        "template": "flux2_schnell",
        "default_width": 1024,
        "default_height": 1024,
    },
    "black-forest-labs/FLUX.2-Fill-dev": {
        "template": "flux2_fill",
        "default_width": 1024,
        "default_height": 1024,
    },
    "black-forest-labs/FLUX.2-Canny-dev": {
        "template": "flux2_canny",
        "default_width": 1024,
        "default_height": 1024,
    },
    "black-forest-labs/FLUX.2-Depth-dev": {
        "template": "flux2_depth",
        "default_width": 1024,
        "default_height": 1024,
    },
    "black-forest-labs/FLUX.2-Redux-dev": {
        "template": "flux2_redux",
        "default_width": 1024,
        "default_height": 1024,
    },
    "black-forest-labs/FLUX.2-Kontext-dev": {
        "template": "flux2_kontext",
        "default_width": 1024,
        "default_height": 1024,
    },
    
    # Wan 2.1
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": {
        "template": "wan_t2v",
        "default_width": 480,
        "default_height": 272,
    },
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": {
        "template": "wan_t2v",
        "default_width": 720,
        "default_height": 480,
    },
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers": {
        "template": "wan_i2v",
        "default_width": 480,
        "default_height": 272,
    },
    "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers": {
        "template": "wan_i2v",
        "default_width": 720,
        "default_height": 480,
    },
    "Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers": {
        "template": "wan_flf2v",
        "default_width": 720,
        "default_height": 480,
    },
    "Wan-AI/Wan2.1-Fun-Control-14B-Diffusers": {
        "template": "wan_fun_control",
        "default_width": 720,
        "default_height": 480,
    },
    
    # QwenImage
    "Qwen/QwenImage-1.5B": {
        "template": "qwen_image_txt2img",
        "default_width": 512,
        "default_height": 512,
    },
}


def resolve_model(model_id: str) -> Tuple[str, Dict[str, Any]]:
    """Resolve a model ID to a template name and kwargs.
    
    Args:
        model_id: HuggingFace model ID or local path.
    
    Returns:
        (template_name, kwargs) tuple.
    
    Raises:
        ValueError: If model ID cannot be resolved.
    """
    # Exact match
    if model_id in MODEL_REGISTRY:
        info = MODEL_REGISTRY[model_id]
        return info["template"], {"pretrained": model_id, **{k: v for k, v in info.items() if k != "template"}}
    
    # Partial match (search by substring)
    model_lower = model_id.lower()
    for registered_id, info in MODEL_REGISTRY.items():
        if registered_id.lower() in model_lower or model_lower in registered_id.lower():
            return info["template"], {"pretrained": model_id, **{k: v for k, v in info.items() if k != "template"}}
    
    # Pattern matching (ordered: more specific first)
    patterns = [
        ("flux.2", "flux2_txt2img"),
        ("flux2", "flux2_txt2img"),
        ("flux.1", "flux_txt2img"),
        ("flux", "flux_txt2img"),
        ("wan2.1", "wan_t2v"),
        ("wan", "wan_t2v"),
        ("sdxl", "sdxl_txt2img"),
        ("sd3", "sd3_txt2img"),
        ("stable-diffusion-3", "sd3_txt2img"),
        ("stable-diffusion-xl", "sdxl_txt2img"),
        ("stable-diffusion", "sd15_txt2img"),
        ("qwenimage", "qwen_image_txt2img"),
        ("qwen", "qwen_image_txt2img"),
    ]
    for pattern, template in patterns:
        if pattern in model_lower:
            return template, {"pretrained": model_id}
    
    raise ValueError(f"Unknown model: '{model_id}'. Register it via hub.MODEL_REGISTRY.")


def get_cache_dir() -> Path:
    """Get the cache directory for downloaded models."""
    cache = os.environ.get("YGGDRASIL_CACHE", None)
    if cache:
        return Path(cache)
    
    hf_cache = os.environ.get("HF_HOME", None)
    if hf_cache:
        return Path(hf_cache) / "yggdrasil"
    
    return Path.home() / ".cache" / "yggdrasil"


def download_model(model_id: str, *, token: Optional[str] = None, force: bool = False) -> Path:
    """Download model weights from HuggingFace Hub.
    
    Args:
        model_id: HuggingFace model ID.
        token: HF token for private models.
        force: Re-download even if cached.
    
    Returns:
        Path to the cached model directory.
    """
    cache_dir = get_cache_dir()
    model_dir = cache_dir / model_id.replace("/", "--")
    
    if model_dir.exists() and not force:
        logger.info(f"Using cached model: {model_dir}")
        return model_dir
    
    try:
        from huggingface_hub import snapshot_download
        
        logger.info(f"Downloading model: {model_id}")
        path = snapshot_download(
            model_id,
            cache_dir=str(cache_dir),
            token=token,
        )
        return Path(path)
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading models. "
            "Install with: pip install huggingface_hub"
        )


def register_model(model_id: str, template: str, **kwargs):
    """Register a new model in the registry.
    
    Args:
        model_id: HuggingFace model ID.
        template: YggDrasil template name.
        **kwargs: Extra config (default_width, default_height, etc.)
    """
    MODEL_REGISTRY[model_id] = {"template": template, **kwargs}
