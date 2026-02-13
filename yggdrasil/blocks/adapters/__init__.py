"""YggDrasil Adapters — все адаптеры автоматически регистрируются."""

from .base import AbstractAdapter
from .lora import LoRAAdapter, DoRAAdapter
from .controlnet import ControlNetAdapter
from .ip_adapter import IPAdapter
from .t2i_adapter import T2IAdapter
from .fusion import CrossAttentionFusionAdapter

# Авто-импорт при загрузке пакета
__all__ = [
    "AbstractAdapter",
    "LoRAAdapter",
    "DoRAAdapter",
    "ControlNetAdapter",
    "IPAdapter",
    "T2IAdapter",
    "CrossAttentionFusionAdapter",
]