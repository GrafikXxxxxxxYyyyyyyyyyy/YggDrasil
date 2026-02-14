"""YggDrasil Block System — сердце Lego-конструктора."""

from .base import AbstractBlock
from .registry import register_block, get_block_class, list_blocks, BlockRegistry
from .slot import Slot
from .port import Port, TensorSpec, InputPort, OutputPort, PortValidator
from .builder import BlockBuilder
from .graph import BlockGraph

__all__ = [
    "AbstractBlock",
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
    "BlockGraph",
]