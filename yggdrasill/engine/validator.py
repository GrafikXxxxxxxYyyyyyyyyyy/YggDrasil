from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from yggdrasill.foundation.node import AbstractGraphNode
from yggdrasill.foundation.port import PortDirection


@dataclass
class ValidationResult:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0


def validate(structure: Any) -> ValidationResult:
    """Validate an executable structure (Hypergraph, Workflow, etc.)."""
    result = ValidationResult()
    node_ids = structure.node_ids
    edges = structure.get_edges()

    for edge in edges:
        if edge.source_node not in node_ids:
            result.errors.append(
                f"Edge references unknown source node '{edge.source_node}'"
            )
            continue
        if edge.target_node not in node_ids:
            result.errors.append(
                f"Edge references unknown target node '{edge.target_node}'"
            )
            continue

        src_node = structure.get_node(edge.source_node)
        dst_node = structure.get_node(edge.target_node)

        src_port = _find_port(src_node, edge.source_port, PortDirection.OUT)
        if src_port is None:
            result.errors.append(
                f"Output port '{edge.source_port}' not found on node '{edge.source_node}'"
            )
        dst_port = _find_port(dst_node, edge.target_port, PortDirection.IN)
        if dst_port is None:
            result.errors.append(
                f"Input port '{edge.target_port}' not found on node '{edge.target_node}'"
            )

        if src_port is not None and dst_port is not None:
            if not src_port.compatible_with(dst_port):
                result.errors.append(
                    f"Incompatible types: {edge.source_node}.{edge.source_port} "
                    f"({src_port.dtype}) -> {edge.target_node}.{edge.target_port} "
                    f"({dst_port.dtype})"
                )

    # Check required inputs are covered
    for nid in node_ids:
        node = structure.get_node(nid)
        if not isinstance(node, AbstractGraphNode):
            continue
        in_edges = structure.get_edges_in(nid)
        covered_ports = {e.target_port for e in in_edges}
        exposed_ports = set()
        for entry in structure.get_input_spec():
            if entry.get("node_id") == nid:
                exposed_ports.add(entry["port_name"])

        for port in node.get_input_ports():
            if not port.optional and port.name not in covered_ports and port.name not in exposed_ports:
                result.errors.append(
                    f"Required input port '{port.name}' on node '{nid}' "
                    f"has no incoming edge and is not an exposed input"
                )

    for entry in structure.get_input_spec():
        nid = entry.get("node_id")
        pname = entry.get("port_name")
        if nid not in node_ids:
            result.errors.append(f"Exposed input references unknown node '{nid}'")
            continue
        node = structure.get_node(nid)
        port = _find_port(node, pname, PortDirection.IN)
        if port is None:
            result.errors.append(
                f"Exposed input port '{pname}' not found as input on node '{nid}'"
            )

    for entry in structure.get_output_spec():
        nid = entry.get("node_id")
        pname = entry.get("port_name")
        if nid not in node_ids:
            result.errors.append(f"Exposed output references unknown node '{nid}'")
            continue
        node = structure.get_node(nid)
        port = _find_port(node, pname, PortDirection.OUT)
        if port is None:
            result.errors.append(
                f"Exposed output port '{pname}' not found as output on node '{nid}'"
            )

    return result


def _find_port(node: Any, port_name: str, direction: PortDirection):
    """Try to find a port on a node, supporting both task-nodes and hypergraph-level nodes."""
    if isinstance(node, AbstractGraphNode):
        port = node.get_port(port_name)
        if port is not None and port.direction == direction:
            return port
        return None

    # For workflow-level nodes (Hypergraph objects), check spec
    if direction == PortDirection.OUT:
        spec = getattr(node, "get_output_spec", lambda **kw: [])(include_dtype=True)
    else:
        spec = getattr(node, "get_input_spec", lambda **kw: [])(include_dtype=True)

    for entry in spec:
        key = entry.get("port_name") or entry.get("name", "")
        if key == port_name:
            from yggdrasill.foundation.port import Port, PortType
            dtype_str = entry.get("dtype", "any")
            try:
                dtype = PortType(dtype_str)
            except ValueError:
                dtype = PortType.ANY
            return Port(name=port_name, direction=direction, dtype=dtype)
    return None
