"""
Pipeline executor: topological order, buffer, run_pipeline(pipeline, inputs, ...) -> outputs.

Canon: WorldGenerator_2.0/TODO_04_PIPELINE.md §4.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from yggdrasill.pipeline.pipeline import Pipeline
from yggdrasill.pipeline.pipeline import _graph_port_key


def _input_key(entry: Dict[str, Any]) -> str:
    return entry.get("name") or f"{entry['pipeline_node_id']}:{entry['port_name']}"


def _output_key(entry: Dict[str, Any]) -> str:
    return entry.get("name") or f"{entry['pipeline_node_id']}:{entry['port_name']}"


def _topological_order(pipeline: Pipeline) -> List[str]:
    in_deg: Dict[str, int] = {nid: 0 for nid in pipeline.node_ids}
    for e in pipeline.get_edges():
        in_deg[e.target_node] = in_deg.get(e.target_node, 0) + 1
    queue = [nid for nid, d in in_deg.items() if d == 0]
    order: List[str] = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for e in pipeline.get_edges_out(nid):
            in_deg[e.target_node] -= 1
            if in_deg[e.target_node] == 0:
                queue.append(e.target_node)
    return order


def run_pipeline(
    pipeline: Pipeline,
    inputs: Dict[str, Any],
    *,
    training: bool = False,
    device: Any = None,
    callbacks: Optional[List[Callable[..., None]]] = None,
    **graph_run_kwargs: Any,
) -> Dict[str, Any]:
    """
    Execute pipeline: run graphs in topological order; buffer (pipeline_node_id, port_key) -> value.
    """
    input_spec = pipeline.get_input_spec()
    buffer: Dict[Tuple[str, str], Any] = {}
    for entry in input_spec:
        key = _input_key(entry)
        if key in inputs:
            buffer[(entry["pipeline_node_id"], entry["port_name"])] = inputs[key]

    if device is not None:
        pipeline.to(device)

    order = _topological_order(pipeline)
    hooks = callbacks or []

    for nid in order:
        g = pipeline.get_graph(nid)
        if g is None:
            continue
        # Gather inputs for this graph from buffer (pipeline edges + exposed)
        graph_inputs: Dict[str, Any] = {}
        for e in pipeline.get_edges_in(nid):
            buf_key = (e.source_node, e.source_port)
            if buf_key in buffer:
                graph_inputs[e.target_port] = buffer[buf_key]
        for spec in g.get_input_spec():
            port_key = _graph_port_key(spec)
            buf_key = (nid, port_key)
            if buf_key in buffer and port_key not in graph_inputs:
                graph_inputs[port_key] = buffer[buf_key]
        for h in hooks:
            try:
                h(nid, "before", inputs=graph_inputs)
            except Exception:
                pass
        out = g.run(graph_inputs, training=training, **graph_run_kwargs)
        for spec in g.get_output_spec():
            port_key = _graph_port_key(spec)
            buffer[(nid, port_key)] = out.get(port_key)
        for h in hooks:
            try:
                h(nid, "after", outputs=out)
            except Exception:
                pass

    result: Dict[str, Any] = {}
    for entry in pipeline.get_output_spec():
        key = (entry["pipeline_node_id"], entry["port_name"])
        out_key = _output_key(entry)
        if key in buffer:
            result[out_key] = buffer[key]
    return result
