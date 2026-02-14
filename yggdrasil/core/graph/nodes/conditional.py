"""ConditionalNode — if/else branching in a graph.

Executes one of two sub-graphs based on a boolean condition.

    cond_node = ConditionalNode.create(
        true_graph=refiner_graph,
        false_graph=passthrough_graph,
    )
    
    # In graph:
    graph.expose_input("use_refiner", "cond_node", "condition")
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from omegaconf import DictConfig

from yggdrasil.core.block.base import AbstractBlock
from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import InputPort, OutputPort, Port


@register_block("graph/conditional")
class ConditionalNode(AbstractBlock):
    """If/else branching: runs true_graph or false_graph based on condition port."""
    
    block_type = "graph/conditional"
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "graph/conditional"}
        super().__init__(config)
        self._true_graph: Optional[Any] = None
        self._false_graph: Optional[Any] = None
    
    @classmethod
    def create(cls, true_graph, false_graph=None) -> ConditionalNode:
        node = cls({"type": "graph/conditional"})
        node._true_graph = true_graph
        node._false_graph = false_graph
        return node
    
    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        return {
            "condition": InputPort("condition", data_type="scalar", description="Boolean condition"),
            "output": OutputPort("output", data_type="any", description="Output from selected branch"),
        }
    
    def process(self, **port_inputs: Any) -> Dict[str, Any]:
        from yggdrasil.core.graph.executor import GraphExecutor
        
        condition = port_inputs.pop("condition", True)
        executor = GraphExecutor(no_grad=True)
        
        if condition and self._true_graph is not None:
            return executor.execute(self._true_graph, **port_inputs)
        elif not condition and self._false_graph is not None:
            return executor.execute(self._false_graph, **port_inputs)
        else:
            # Passthrough
            return port_inputs
    
    def _forward_impl(self, **kwargs):
        return self.process(**kwargs)
    
    def __repr__(self):
        t = len(self._true_graph.nodes) if self._true_graph else 0
        f = len(self._false_graph.nodes) if self._false_graph else 0
        return f"<ConditionalNode true_nodes={t} false_nodes={f}>"
