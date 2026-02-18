# yggdrasil/core/unified/modality.py
"""Модальности и типы сигналов — единый словарь для любого диффузионного пайплайна.

Любой пайплайн (изображение, видео, аудио, молекулы, 3D) оперирует:
- сигналом в «сыром» пространстве (pixel, waveform, point cloud, ...),
- латентным представлением,
- опциональными условиями и управляющими сигналами.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple


class Modality(str, Enum):
    """Модальность генерируемого контента."""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    SPEECH = "speech"
    MUSIC = "music"
    MOLECULAR = "molecular"
    STRUCTURE_3D = "structure_3d"
    TIMESERIES = "timeseries"
    ANY = "any"


class SignalSpace(str, Enum):
    """Пространство представления сигнала (совместимо с TensorSpec.space)."""
    PIXEL = "pixel"
    LATENT = "latent"
    EMBEDDING = "embedding"
    NOISE = "noise"
    WAVEFORM = "waveform"
    POINT_CLOUD = "point_cloud"
    GRAPH = "graph"       # для молекул / графов
    SCALAR = "scalar"
    ANY = "any"


# Соответствие модальность -> типичные пространства входа/выхода
MODALITY_DEFAULT_SPACES: Dict[Modality, Tuple[str, str]] = {
    Modality.IMAGE: ("pixel", "latent"),
    Modality.VIDEO: ("pixel", "latent"),
    Modality.AUDIO: ("waveform", "latent"),
    Modality.SPEECH: ("waveform", "latent"),
    Modality.MUSIC: ("waveform", "latent"),
    Modality.MOLECULAR: ("graph", "latent"),
    Modality.STRUCTURE_3D: ("point_cloud", "latent"),
    Modality.TIMESERIES: ("waveform", "latent"),
    Modality.ANY: ("any", "any"),
}


def get_modality_from_spec(spec: Any) -> Modality:
    """По TensorSpec.modality или строке вернуть Modality."""
    if spec is None:
        return Modality.ANY
    modality = getattr(spec, "modality", None) or (spec if isinstance(spec, str) else None)
    if modality is None or modality == "any":
        return Modality.ANY
    try:
        return Modality(modality)
    except ValueError:
        return Modality.ANY
