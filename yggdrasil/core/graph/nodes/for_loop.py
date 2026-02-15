"""ForLoopNode — generic iteration over an inner graph.

NOT diffusion-specific. Works for any iterative process.

    loop = ForLoopNode(
        inner_graph=step_graph,
        num_iterations=28,
        carry_vars=["x"],
        schedule=my_schedule_tensor,   # optional
    )

At each iteration:
    - carry variables are passed from previous output to next input
    - step_index (int) is available as an input
    - if schedule is provided, schedule[i] is passed as 'schedule_value'
    - all other inputs are passed through unchanged
"""
from __future__ import annotations

import torch
from collections import OrderedDict
from typing import Any, Dict, List, Optional
from omegaconf import DictConfig

from yggdrasil.core.block.base import AbstractBaseBlock
from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import InputPort, OutputPort, Port


@register_block("graph/for_loop")
class ForLoopNode(AbstractBaseBlock):
    """Generic for-loop: executes inner graph N times with carry variables."""
    
    block_type = "graph/for_loop"
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "graph/for_loop"}
        super().__init__(config)
        self.num_iterations: int = int(self.config.get("num_iterations", 1))
        self.carry_vars: List[str] = list(self.config.get("carry_vars", ["x"]))
        self.show_progress: bool = bool(self.config.get("show_progress", True))
        self._inner_graph: Optional[Any] = None
    
    @property
    def graph(self):
        return self._inner_graph
    
    @graph.setter
    def graph(self, value):
        self._inner_graph = value
    
    @classmethod
    def create(
        cls,
        inner_graph,
        num_iterations: int = 1,
        carry_vars: Optional[List[str]] = None,
        show_progress: bool = True,
        schedule: Optional[torch.Tensor] = None,
    ) -> ForLoopNode:
        """Factory method."""
        node = cls({
            "type": "graph/for_loop",
            "num_iterations": num_iterations,
            "carry_vars": carry_vars or ["x"],
            "show_progress": show_progress,
        })
        node._inner_graph = inner_graph
        node._schedule = schedule
        return node
    
    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        """Dynamic: depends on inner graph. Base ports shown here."""
        return {
            "initial_x": InputPort("initial_x", description="Initial carry variable"),
            "schedule": InputPort("schedule", data_type="tensor", optional=True, description="Iteration schedule"),
            "x": OutputPort("x", description="Final carry variable"),
        }
    
    def get_dynamic_ports(self) -> Dict[str, Port]:
        """Get ports based on inner graph's exposed I/O."""
        ports = {}
        if self._inner_graph is not None:
            for name, targets in self._inner_graph.graph_inputs.items():
                if name not in self.carry_vars:
                    ports[name] = InputPort(name, data_type="any", optional=True,
                                            description=f"Passthrough input -> inner.{name}")
            for name, _ in self._inner_graph.graph_outputs.items():
                if name not in self.carry_vars:
                    ports[name] = OutputPort(name, data_type="any",
                                             description=f"Final output from inner.{name}")
        return {**self.declare_io(), **ports}
    
    def process(self, **port_inputs: Any) -> Dict[str, Any]:
        """Execute the inner graph num_iterations times."""
        from yggdrasil.core.graph.executor import GraphExecutor
        
        if self._inner_graph is None:
            return {}
        
        # Initialize carry variables
        carry = {}
        for var in self.carry_vars:
            initial_key = f"initial_{var}"
            carry[var] = port_inputs.get(initial_key, port_inputs.get(var))
        
        # Schedule
        schedule = port_inputs.get("schedule", getattr(self, '_schedule', None))
        
        executor = GraphExecutor(no_grad=True)
        
        iterator = range(self.num_iterations)
        if self.show_progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(iterator, total=self.num_iterations, desc="ForLoop")
            except ImportError:
                pass
        
        last_outputs = {}
        for i in iterator:
            step_inputs = dict(carry)
            step_inputs["step_index"] = i
            
            if schedule is not None and isinstance(schedule, torch.Tensor) and i < len(schedule):
                step_inputs["schedule_value"] = schedule[i]
            
            # Pass through non-carry, non-special inputs
            for k, v in port_inputs.items():
                if k not in carry and not k.startswith("initial_") and k != "schedule":
                    step_inputs[k] = v
            
            last_outputs = executor.execute(self._inner_graph, **step_inputs)
            
            # Update carry from outputs
            for var in self.carry_vars:
                if var in last_outputs:
                    carry[var] = last_outputs[var]
                elif f"next_{var}" in last_outputs:
                    carry[var] = last_outputs[f"next_{var}"]
        
        # Final result: carry vars + last outputs
        result = dict(last_outputs)
        result.update(carry)
        return result
    
    def _forward_impl(self, **kwargs):
        return self.process(**kwargs)
    
    def __repr__(self):
        n = len(self._inner_graph.nodes) if self._inner_graph else 0
        return f"<ForLoopNode iters={self.num_iterations} carry={self.carry_vars} inner_nodes={n}>"
