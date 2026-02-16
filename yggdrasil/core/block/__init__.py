"""YggDrasil Block System — сердце Lego-конструктора."""

from .base import AbstractBaseBlock
from .registry import register_block, get_block_class, list_blocks, BlockRegistry
from .slot import Slot
from .port import Port, TensorSpec, InputPort, OutputPort, PortValidator
from .builder import BlockBuilder

__all__ = [
    "AbstractBaseBlock",
    "register_block",
    "get_block_class",
    "list_blocks",
    "BlockRegistry",
    "Slot",
    "Port",
    "TensorSpec",
    "InputPort",
    "OutputPort",
    "PortValidator",
    "BlockBuilder",
]