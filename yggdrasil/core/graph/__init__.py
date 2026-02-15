"""YggDrasil Compute Graph — настоящий dataflow Lego-конструктор.

ComputeGraph — DAG из блоков, соединённых через типизированные порты.
Это главная структура данных для сборки произвольных диффузионных pipeline.
"""

from .graph import ComputeGraph, Edge
from .executor import GraphExecutor
from .subgraph import SubGraph
from .stage import AbstractStage
from .adapters import add_controlnet_to_graph, add_ip_adapter_to_graph, add_adapter_to_graph

__all__ = [
    "AbstractStage",
    "ComputeGraph",
    "Edge",
    "GraphExecutor",
    "SubGraph",
    "add_controlnet_to_graph",
    "add_ip_adapter_to_graph",
    "add_adapter_to_graph",
]
