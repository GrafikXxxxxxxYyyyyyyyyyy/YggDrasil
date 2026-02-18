"""Text modality plugin for YggDrasil.

Supports: Diffusion-LM, MDLM, discrete diffusion for text generation.
"""
from .plugin import TextPlugin

TextPlugin.register_blocks()
