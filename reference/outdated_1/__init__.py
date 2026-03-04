"""
Yggdrasil — universal Lego framework for diffusion and generative models.

Canon: WorldGenerator_2.0 (Scheme, Abstract_Block_And_Node, TODOs).
Levels: foundation → task_nodes → graph → pipeline → stage → world.
"""

__version__ = "0.3.0"

from yggdrasill.foundation import (
    Port,
    PortDirection,
    PortAggregation,
    PortType,
    AbstractBaseBlock,
    Node,
    Edge,
    Graph,
    ValidationResult,
    BlockRegistry,
)
from yggdrasill.executor import run as run_graph
from yggdrasill.pipeline import Pipeline, PipelineEdge

__all__ = [
    "__version__",
    "Port",
    "PortDirection",
    "PortAggregation",
    "PortType",
    "AbstractBaseBlock",
    "Node",
    "Edge",
    "Graph",
    "ValidationResult",
    "BlockRegistry",
    "run_graph",
    "Pipeline",
    "PipelineEdge",
]
