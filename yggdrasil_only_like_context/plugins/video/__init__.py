"""Video modality plugin for YggDrasil.

Supports: CogVideoX, Hunyuan Video, Mochi, and custom video diffusion models.
"""
from .plugin import VideoPlugin

VideoPlugin.register_blocks()
