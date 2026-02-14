"""Molecular modality plugin for YggDrasil.

Supports: DiffDock, GeoLDM, EquiFold, and custom molecular diffusion.
"""
from .plugin import MolecularPlugin

MolecularPlugin.register_blocks()
