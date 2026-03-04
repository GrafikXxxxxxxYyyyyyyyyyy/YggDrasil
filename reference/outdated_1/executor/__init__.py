"""
Graph executor: run(graph, inputs, ...) -> outputs.

Canon: WorldGenerator_2.0/TODO_03_GRAPH_ENGINE.md §4.
- Topological order + SCC for cycles; buffer (node_id, port_name) -> value.
- run(..., training=False, num_loop_steps=None, device=None, callbacks=None).
"""

from yggdrasill.executor.run import run

__all__ = ["run"]
