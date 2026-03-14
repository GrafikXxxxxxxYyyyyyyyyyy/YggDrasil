from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from yggdrasill.engine.buffers import EdgeBuffers
from yggdrasill.engine.planner import build_plan
from yggdrasill.engine.validator import validate
from yggdrasill.foundation.node import AbstractGraphNode
from yggdrasill.foundation.port import PortAggregation


class ValidationError(Exception):
    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__(f"Validation failed: {errors}")


def run(
    structure: Any,
    inputs: Dict[str, Any],
    *,
    training: bool = False,
    num_loop_steps: Optional[int] = None,
    device: Optional[Any] = None,
    callbacks: Optional[List[Callable[..., None]]] = None,
    dry_run: bool = False,
    validate_before: bool = True,
) -> Dict[str, Any]:
    """Execute the structure (Hypergraph, Workflow, etc.) and return outputs."""
    if validate_before:
        result = validate(structure)
        if not result.valid:
            raise ValidationError(result.errors)

    K = num_loop_steps
    if K is None:
        meta = getattr(structure, "metadata", {}) or {}
        K = meta.get("num_loop_steps", 1)

    input_spec = structure.get_input_spec()
    output_spec = structure.get_output_spec()

    buf = EdgeBuffers.init_from_inputs(input_spec, inputs)

    _prepare_nodes(structure, training, device)

    plan = build_plan(structure)

    cbs = callbacks or []

    for step_type, step_data in plan:
        if step_type == "node":
            _execute_node(structure, step_data, buf, input_spec, dry_run, cbs)
        elif step_type == "cycle":
            rep, comp = step_data
            node_order = sorted(comp)
            _fire_callbacks(cbs, "loop_start", {"nodes": node_order, "steps": K})
            for _iteration in range(K):
                for nid in node_order:
                    _execute_node(structure, nid, buf, input_spec, dry_run, cbs)
            _fire_callbacks(cbs, "loop_end", {"nodes": node_order, "steps": K})

    outputs: Dict[str, Any] = {}
    for entry in output_spec:
        key = _spec_key(entry)
        nid = entry.get("node_id") or entry.get("graph_id")
        pname = entry["port_name"]
        outputs[key] = buf.read(nid, pname)
    return outputs


def _prepare_nodes(structure: Any, training: bool, device: Any) -> None:
    for nid in structure.node_ids:
        node = structure.get_node(nid)
        if node is None:
            continue
        if hasattr(node, "train") and callable(node.train):
            node.train(training)
        if device is not None and hasattr(node, "to") and callable(node.to):
            node.to(device)


def _execute_node(
    structure: Any,
    node_id: str,
    buf: EdgeBuffers,
    input_spec: List[Dict[str, Any]],
    dry_run: bool,
    callbacks: List[Callable[..., None]],
) -> None:
    node = structure.get_node(node_id)
    if node is None:
        return

    node_inputs: Dict[str, Any] = {}

    in_edges = structure.get_edges_in(node_id)

    port_edge_counts: Dict[str, int] = {}
    for edge in in_edges:
        port_edge_counts[edge.target_port] = port_edge_counts.get(edge.target_port, 0) + 1

    for target_port, count in port_edge_counts.items():
        if count == 1:
            edge = next(e for e in in_edges if e.target_port == target_port)
            if buf.has(edge.source_node, edge.source_port):
                node_inputs[target_port] = buf.read(edge.source_node, edge.source_port)
        else:
            agg_policy = _get_aggregation(node, target_port)
            buf.clear_multi(node_id, target_port)
            for edge in in_edges:
                if edge.target_port != target_port:
                    continue
                if buf.has(edge.source_node, edge.source_port):
                    buf.append(
                        node_id, target_port,
                        buf.read(edge.source_node, edge.source_port),
                        source_node=edge.source_node,
                    )
            if buf.has_multi(node_id, target_port):
                node_inputs[target_port] = buf.aggregate(node_id, target_port, agg_policy)

    for entry in input_spec:
        nid = entry.get("node_id") or entry.get("graph_id")
        if nid == node_id:
            pname = entry["port_name"]
            if pname not in node_inputs:
                if buf.has(nid, pname):
                    node_inputs[pname] = buf.read(nid, pname)

    _fire_callbacks(callbacks, "before", {"node_id": node_id})

    if dry_run:
        node_outputs: Dict[str, Any] = {}
        if hasattr(node, "get_output_ports"):
            for port in node.get_output_ports():
                node_outputs[port.name] = None
        elif hasattr(node, "get_output_spec"):
            for entry in node.get_output_spec():
                node_outputs[entry.get("name") or entry["port_name"]] = None
    else:
        node_outputs = node.run(node_inputs)

    for port_name, value in node_outputs.items():
        buf.write(node_id, port_name, value)

    _fire_callbacks(callbacks, "after", {"node_id": node_id})


def _get_aggregation(node: Any, port_name: str) -> PortAggregation:
    """Retrieve the aggregation policy for *port_name* on *node*."""
    if isinstance(node, AbstractGraphNode):
        port = node.get_port(port_name)
        if port is not None:
            return port.aggregation
    return PortAggregation.SINGLE


def _spec_key(entry: Dict[str, Any]) -> str:
    if "name" in entry and entry["name"] is not None:
        return entry["name"]
    nid = entry.get("node_id") or entry.get("graph_id", "")
    return f"{nid}:{entry['port_name']}"


def _fire_callbacks(
    callbacks: List[Callable[..., None]], phase: str, info: Dict[str, Any],
) -> None:
    for cb in callbacks:
        try:
            cb(phase, info)
        except Exception:
            pass
