from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from yggdrasill.engine.planner import build_plan
from yggdrasill.engine.validator import validate


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
        K = structure.metadata.get("num_loop_steps", 1)

    input_spec = structure.get_input_spec()
    output_spec = structure.get_output_spec()

    buffer: Dict[tuple, Any] = {}
    for entry in input_spec:
        key = _spec_key(entry)
        nid = entry["node_id"] if "node_id" in entry else entry.get("graph_id")
        pname = entry["port_name"]
        if key in inputs:
            buffer[(nid, pname)] = inputs[key]
        elif (nid, pname) in inputs:
            buffer[(nid, pname)] = inputs[(nid, pname)]

    plan = build_plan(structure)

    cbs = callbacks or []

    for step_type, step_data in plan:
        if step_type == "node":
            _execute_node(
                structure, step_data, buffer, input_spec, dry_run, cbs, training, device,
            )
        elif step_type == "cycle":
            rep, comp = step_data
            node_order = sorted(comp)
            _fire_callbacks(cbs, "loop_start", {"nodes": node_order, "steps": K})
            for _iteration in range(K):
                for nid in node_order:
                    _execute_node(
                        structure, nid, buffer, input_spec, dry_run, cbs, training, device,
                    )
            _fire_callbacks(cbs, "loop_end", {"nodes": node_order, "steps": K})

    outputs: Dict[str, Any] = {}
    for entry in output_spec:
        key = _spec_key(entry)
        nid = entry["node_id"] if "node_id" in entry else entry.get("graph_id")
        pname = entry["port_name"]
        outputs[key] = buffer.get((nid, pname))
    return outputs


def _execute_node(
    structure: Any,
    node_id: str,
    buffer: Dict[tuple, Any],
    input_spec: List[Dict[str, Any]],
    dry_run: bool,
    callbacks: List[Callable[..., None]],
    training: bool,
    device: Any,
) -> None:
    node = structure.get_node(node_id)
    if node is None:
        return

    if hasattr(node, "train") and callable(node.train):
        node.train(training)
    if device is not None and hasattr(node, "to") and callable(node.to):
        node.to(device)

    # Gather inputs for this node
    node_inputs: Dict[str, Any] = {}

    in_edges = structure.get_edges_in(node_id)
    for edge in in_edges:
        val = buffer.get((edge.source_node, edge.source_port))
        if val is not None:
            node_inputs[edge.target_port] = val

    for entry in input_spec:
        nid = entry.get("node_id") or entry.get("graph_id")
        if nid == node_id:
            pname = entry["port_name"]
            if pname not in node_inputs:
                buf_val = buffer.get((nid, pname))
                if buf_val is not None:
                    node_inputs[pname] = buf_val

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
        buffer[(node_id, port_name)] = value

    _fire_callbacks(callbacks, "after", {"node_id": node_id})


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
