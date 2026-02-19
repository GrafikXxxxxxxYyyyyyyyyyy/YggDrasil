"""
Pipeline: graph of graphs (TODO_04).

Canon: WorldGenerator_2.0/TODO_04_PIPELINE.md, Pipeline_Level.md.
Nodes = Graph instances; edges connect graph external ports; run(pipeline, inputs) -> outputs.
"""

from yggdrasill.pipeline.pipeline import Pipeline, PipelineEdge

__all__ = ["Pipeline", "PipelineEdge"]
