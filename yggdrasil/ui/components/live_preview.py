"""Live preview component using streaming generation."""
from __future__ import annotations

import torch
import numpy as np
from typing import Optional, Generator, Any, Dict


def decode_latents_to_preview(
    latents: torch.Tensor,
    model=None,
) -> np.ndarray:
    """Decode latents to a preview image array.
    
    Args:
        latents: Latent tensor [B, C, H, W]
        model: ModularDiffusionModel with codec
        
    Returns:
        numpy array [H, W, 3] suitable for Gradio Image component
    """
    if model is not None and hasattr(model, "decode"):
        with torch.no_grad():
            image = model.decode(latents)
            image = (image / 2 + 0.5).clamp(0, 1)
    else:
        # Simple visualization of latents
        image = latents[:, :3]  # Take first 3 channels
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    image = image[0].cpu().permute(1, 2, 0).numpy()
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return image


def streaming_generate(
    sampler,
    condition: Dict[str, Any],
    **kwargs,
) -> Generator[np.ndarray, None, None]:
    """Generator that yields preview images during sampling.
    
    Args:
        sampler: DiffusionSampler instance
        condition: Generation condition dict
        
    Yields:
        numpy arrays of intermediate results
    """
    for intermediate in sampler.sample_iter(condition=condition, **kwargs):
        image = intermediate[0].cpu()
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).clip(0, 255).astype(np.uint8)
        yield image
