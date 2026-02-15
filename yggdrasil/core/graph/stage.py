# yggdrasil/core/graph/stage.py
"""AbstractStage — одна стадия пайплайна.

Стадия = граф блоков уровня 1. Сама AbstractStage выступает узлом в графе пайплайна.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from omegaconf import OmegaConf
from pathlib import Path
import torch.nn as nn

from ..block.base import AbstractBaseBlock
from ..block.registry import register_block
from ..block.port import Port, InputPort, OutputPort
from .graph import ComputeGraph, Edge
from .executor import GraphExecutor


@register_block("stage/abstract")
class AbstractStage(AbstractBaseBlock):
    """Одна стадия пайплайна — граф блоков уровня 1.

    AbstractStage содержит ComputeGraph из блоков (Backbone, Codec, Conditioner,
    Guidance, Solver, Adapter, InnerModule, OuterModule, Processor).
    Сама AbstractStage выступает узлом в графе пайплайна (InferencePipeline).
    Prior-, Detailer-, Upscaler-пайплайны задаются как AbstractStage.
    """

    block_type = "stage/abstract"

    def __init__(self, config=None, graph: Optional[ComputeGraph] = None):
        config = config or {"type": "stage/abstract"}
        super().__init__(config)
        self.graph: ComputeGraph = graph or ComputeGraph(name=getattr(self, "block_id", "stage"))

    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        return {
            "input": InputPort("input", data_type="any", optional=True, description="Stage input"),
            "output": OutputPort("output", data_type="any", description="Stage output"),
        }

    def process(self, **port_inputs) -> Dict[str, Any]:
        """Execute inner graph and return outputs. Ensures 'output' key for pipeline chaining."""
        outputs = GraphExecutor().execute(self.graph, **port_inputs)
        if "output" not in outputs and outputs:
            # For pipeline links (stage0.output -> stage1.input), expose first output as "output"
            first_key = next(iter(outputs))
            outputs["output"] = outputs[first_key]
        return outputs

    def add_node(self, name: str, block: AbstractBaseBlock) -> "AbstractStage":
        """Добавить узел в граф стадии."""
        self.graph.add_node(name, block)
        return self

    def connect(self, src: str, src_port: str, dst: str, dst_port: str) -> "AbstractStage":
        """Соединить порты в графе стадии."""
        self.graph.connect(src, src_port, dst, dst_port)
        return self

    def expose_input(self, graph_input_name: str, node: str, port: str) -> "AbstractStage":
        """Объявить вход графа стадии."""
        self.graph.expose_input(graph_input_name, node, port)
        return self

    def expose_output(self, graph_output_name: str, node: str, port: str) -> "AbstractStage":
        """Объявить выход графа стадии."""
        self.graph.expose_output(graph_output_name, node, port)
        return self

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Yield parameters from the inner graph (for training pipeline stages)."""
        for block in self.graph.nodes.values():
            if hasattr(block, "parameters"):
                yield from block.parameters(recurse=recurse)

    def state_dict(self, *args, **kwargs):
        """State dict of the inner graph (for checkpointing when stage is a train node)."""
        return {
            "_stage_graph": {
                n: (b.state_dict(*args, **kwargs) if hasattr(b, "state_dict") else {})
                for n, b in self.graph.nodes.items()
            }
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load state into the inner graph."""
        inner = state_dict.get("_stage_graph", state_dict)
        for n, state in inner.items():
            if n in self.graph.nodes and state and hasattr(self.graph.nodes[n], "load_state_dict"):
                self.graph.nodes[n].load_state_dict(state, strict=strict)
