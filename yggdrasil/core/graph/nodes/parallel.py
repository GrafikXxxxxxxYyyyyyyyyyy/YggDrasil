"""ParallelNode — execute multiple sub-graphs with the same inputs.

Results are merged into a single dict (or concatenated).

    parallel = ParallelNode.create(
        branches={"cond": cond_graph, "uncond": uncond_graph},
        merge_strategy="dict",  # or "concat" along dim=0
    )
"""
from __future__ import annotations

import torch
from typing import Any, Dict, Optional
from omegaconf import DictConfig

from yggdrasil.core.block.base import AbstractBlock
from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import InputPort, OutputPort, Port


@register_block("graph/parallel")
class ParallelNode(AbstractBlock):
    """Execute branches in parallel (sequentially in practice) and merge outputs."""
    
    block_type = "graph/parallel"
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "graph/parallel"}
        super().__init__(config)
        self._branches: Dict[str, Any] = {}
        self.merge_strategy: str = self.config.get("merge_strategy", "dict")
    
    @classmethod
    def create(cls, branches: Dict[str, Any], merge_strategy: str = "dict") -> ParallelNode:
        node = cls({"type": "graph/parallel", "merge_strategy": merge_strategy})
        node._branches = branches
        return node
    
    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        return {
            "output": OutputPort("output", data_type="any", description="Merged output from all branches"),
        }
    
    def process(self, **port_inputs: Any) -> Dict[str, Any]:
        from yggdrasil.core.graph.executor import GraphExecutor
        
        executor = GraphExecutor(no_grad=True)
        branch_outputs = {}
        
        for name, graph in self._branches.items():
            if graph is not None:
                branch_outputs[name] = executor.execute(graph, **port_inputs)
        
        if self.merge_strategy == "concat":
            # Concatenate tensor outputs along batch dim
            merged = {}
            for key in branch_outputs.get(list(branch_outputs.keys())[0], {}).keys():
                tensors = []
                for name in branch_outputs:
                    val = branch_outputs[name].get(key)
                    if isinstance(val, torch.Tensor):
                        tensors.append(val)
                if tensors:
                    merged[key] = torch.cat(tensors, dim=0)
            merged["output"] = merged.get("output", next(iter(merged.values()), None))
            return merged
        else:
            # Dict strategy: prefix outputs with branch name
            result = {}
            for name, outputs in branch_outputs.items():
                for key, val in outputs.items():
                    result[f"{name}_{key}"] = val
            result["output"] = branch_outputs
            return result
    
    def _forward_impl(self, **kwargs):
        return self.process(**kwargs)
    
    def __repr__(self):
        return f"<ParallelNode branches={list(self._branches.keys())} merge={self.merge_strategy}>"
