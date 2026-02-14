"""3D modality plugin for YggDrasil.

Supports: Gaussian Splatting, Point Cloud, Mesh diffusion.
"""
from .plugin import ThreeDPlugin

ThreeDPlugin.register_blocks()
