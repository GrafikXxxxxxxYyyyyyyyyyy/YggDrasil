"""Image modality plugin for YggDrasil.

Supports: SD 1.5, SDXL, SD3, Flux, and custom image diffusion models.
"""
from .plugin import ImagePlugin

# Auto-register blocks on import
ImagePlugin.register_blocks()
