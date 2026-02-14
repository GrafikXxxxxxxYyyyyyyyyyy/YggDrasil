"""Generic graph primitives: ForLoop, Conditional, Parallel."""
from .for_loop import ForLoopNode
from .conditional import ConditionalNode
from .parallel import ParallelNode

__all__ = ["ForLoopNode", "ConditionalNode", "ParallelNode"]
