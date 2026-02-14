"""YggDrasil Compute Graph — настоящий dataflow Lego-конструктор.

ComputeGraph — DAG из блоков, соединённых через типизированные порты.
Это главная структура данных для сборки произвольных диффузионных pipeline.
"""

from .graph import ComputeGraph, Edge
from .executor import GraphExecutor
from .subgraph import SubGraph

__all__ = [
    "ComputeGraph",
    "Edge",
    "GraphExecutor",
    "SubGraph",
]
