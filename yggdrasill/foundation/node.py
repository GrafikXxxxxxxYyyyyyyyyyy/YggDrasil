"""
Abstract Graph Node: position in graph + reference to block.

Canon: WorldGenerator_2.0/Abstract_Block_And_Node.md §3, TODO_01 §3.
- node_id (unique in graph), block reference
- Edges stored at graph level; node provides block and identity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from yggdrasill.foundation.block import AbstractBaseBlock

if TYPE_CHECKING:
    from yggdrasill.foundation.graph import Edge


class Node:
    """
    Place in the graph: unique node_id and exactly one block.
    Does not store data; edges are maintained by Graph.
    """

    __slots__ = ("_node_id", "_block")

    def __init__(self, node_id: str, block: AbstractBaseBlock) -> None:
        if not node_id or not node_id.strip():
            raise ValueError("node_id must be non-empty")
        self._node_id = node_id.strip()
        self._block = block

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def block(self) -> AbstractBaseBlock:
        return self._block

    def get_input_ports(self):
        return self._block.get_input_ports()

    def get_output_ports(self):
        return self._block.get_output_ports()

    def __repr__(self) -> str:
        return f"Node(node_id={self.node_id!r}, block={self.block})"
