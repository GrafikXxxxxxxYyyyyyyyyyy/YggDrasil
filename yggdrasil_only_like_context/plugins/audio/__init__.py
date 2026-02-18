"""Audio modality plugin for YggDrasil.

Supports: AudioLDM, Stable Audio, MusicGen-style architectures.
"""
from .plugin import AudioPlugin

AudioPlugin.register_blocks()
