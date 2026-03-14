"""Workflow: a hypergraph of hypergraphs.

Nodes are Hypergraph instances.  Edges connect their exposed outputs
to other hypergraphs' exposed inputs.  The same engine (Validator,
Planner, Executor) runs the workflow without any changes.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from yggdrasill.engine.edge import Edge
from yggdrasill.engine.structure import Hypergraph


class Workflow:
    """Hypergraph of hypergraphs -- implements the same structural protocol
    that the engine expects (node_ids, get_node, get_edges, etc.)."""

    def __init__(self, workflow_id: Optional[str] = None) -> None:
        self._workflow_id = workflow_id or "workflow"
        self._metadata: Dict[str, Any] = {}

        self._nodes: Dict[str, Hypergraph] = {}
        self._edges: List[Edge] = []
        self._in_edges: Dict[str, List[Edge]] = {}
        self._out_edges: Dict[str, List[Edge]] = {}

        self._exposed_inputs: List[Dict[str, Any]] = []
        self._exposed_outputs: List[Dict[str, Any]] = []

        self._execution_version: int = 0

    # --- identity / metadata (same interface as Hypergraph) ---------------

    @property
    def graph_id(self) -> str:
        return self._workflow_id

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        self._metadata = dict(value)

    @property
    def execution_version(self) -> int:
        return self._execution_version

    # --- structural protocol (engine-facing) ------------------------------

    @property
    def node_ids(self) -> Set[str]:
        return set(self._nodes.keys())

    def get_node(self, node_id: str) -> Optional[Hypergraph]:
        return self._nodes.get(node_id)

    def get_edges(self) -> List[Edge]:
        return list(self._edges)

    def get_edges_in(self, node_id: str) -> List[Edge]:
        return list(self._in_edges.get(node_id, []))

    def get_edges_out(self, node_id: str) -> List[Edge]:
        return list(self._out_edges.get(node_id, []))

    def get_input_spec(self, include_dtype: bool = False) -> List[Dict[str, Any]]:
        return [dict(e) for e in self._exposed_inputs]

    def get_output_spec(self, include_dtype: bool = False) -> List[Dict[str, Any]]:
        return [dict(e) for e in self._exposed_outputs]

    # --- node management --------------------------------------------------

    def add_node(self, node_id: str, hypergraph: Hypergraph) -> None:
        if not node_id or not node_id.strip():
            raise ValueError("node_id must be non-empty")
        node_id = node_id.strip()
        self._nodes[node_id] = hypergraph
        self._in_edges.setdefault(node_id, [])
        self._out_edges.setdefault(node_id, [])
        self._execution_version += 1

    def remove_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            return
        del self._nodes[node_id]
        self._edges = [
            e for e in self._edges
            if e.source_node != node_id and e.target_node != node_id
        ]
        self._in_edges.pop(node_id, None)
        self._out_edges.pop(node_id, None)
        for nid in list(self._in_edges):
            self._in_edges[nid] = [
                e for e in self._in_edges[nid] if e.source_node != node_id
            ]
        for nid in list(self._out_edges):
            self._out_edges[nid] = [
                e for e in self._out_edges[nid] if e.target_node != node_id
            ]
        self._exposed_inputs = [
            ei for ei in self._exposed_inputs if ei.get("node_id") != node_id
        ]
        self._exposed_outputs = [
            eo for eo in self._exposed_outputs if eo.get("node_id") != node_id
        ]
        self._execution_version += 1

    # --- edge management --------------------------------------------------

    def add_edge(self, edge: Edge) -> None:
        if edge.source_node not in self._nodes:
            raise ValueError(f"Source node '{edge.source_node}' not in workflow")
        if edge.target_node not in self._nodes:
            raise ValueError(f"Target node '{edge.target_node}' not in workflow")
        if edge in self._edges:
            return
        self._edges.append(edge)
        self._in_edges.setdefault(edge.target_node, []).append(edge)
        self._out_edges.setdefault(edge.source_node, []).append(edge)
        self._execution_version += 1

    # --- exposed ports ----------------------------------------------------

    def expose_input(
        self, node_id: str, port_name: str, name: Optional[str] = None,
    ) -> None:
        if node_id not in self._nodes:
            raise ValueError(f"Node '{node_id}' not in workflow")
        entry: Dict[str, Any] = {"node_id": node_id, "port_name": port_name}
        if name is not None:
            entry["name"] = name
        for existing in self._exposed_inputs:
            if existing.get("node_id") == node_id and existing.get("port_name") == port_name:
                return
        self._exposed_inputs.append(entry)
        self._execution_version += 1

    def expose_output(
        self, node_id: str, port_name: str, name: Optional[str] = None,
    ) -> None:
        if node_id not in self._nodes:
            raise ValueError(f"Node '{node_id}' not in workflow")
        entry: Dict[str, Any] = {"node_id": node_id, "port_name": port_name}
        if name is not None:
            entry["name"] = name
        for existing in self._exposed_outputs:
            if existing.get("node_id") == node_id and existing.get("port_name") == port_name:
                return
        self._exposed_outputs.append(entry)
        self._execution_version += 1

    # --- run (delegates to engine) ----------------------------------------

    def run(
        self,
        inputs: Dict[str, Any],
        *,
        training: bool = False,
        num_loop_steps: Optional[int] = None,
        device: Optional[Any] = None,
        callbacks: Optional[list] = None,
        dry_run: bool = False,
        validate_before: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from yggdrasill.engine.executor import run as _run
        return _run(
            self, inputs,
            training=training,
            num_loop_steps=num_loop_steps,
            device=device,
            callbacks=callbacks,
            dry_run=dry_run,
            validate_before=validate_before,
        )

    # --- state dict -------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for nid, hg in self._nodes.items():
            sd = hg.state_dict()
            if sd:
                result[nid] = sd
        return result

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        for nid, hg in self._nodes.items():
            if nid in state:
                hg.load_state_dict(state[nid], strict=strict)

    # --- config -----------------------------------------------------------

    def to_config(self) -> Dict[str, Any]:
        nodes = []
        for nid, hg in self._nodes.items():
            nodes.append({
                "node_id": nid,
                "hypergraph_config": hg.to_config(),
            })

        edges = [
            {
                "source_node": e.source_node,
                "source_port": e.source_port,
                "target_node": e.target_node,
                "target_port": e.target_port,
            }
            for e in self._edges
        ]

        cfg: Dict[str, Any] = {
            "schema_version": "1.0",
            "workflow_id": self._workflow_id,
            "nodes": nodes,
            "edges": edges,
            "exposed_inputs": [dict(ei) for ei in self._exposed_inputs],
            "exposed_outputs": [dict(eo) for eo in self._exposed_outputs],
        }
        if self._metadata:
            cfg["metadata"] = dict(self._metadata)
        return cfg

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        *,
        registry: Optional[Any] = None,
    ) -> "Workflow":
        w = cls(workflow_id=config.get("workflow_id", "workflow"))
        w.metadata = dict(config.get("metadata", {}))

        for nc in config.get("nodes", []):
            nid = nc["node_id"]
            hg = Hypergraph.from_config(nc["hypergraph_config"], registry=registry)
            w.add_node(nid, hg)

        for ec in config.get("edges", []):
            w.add_edge(Edge(
                source_node=ec["source_node"],
                source_port=ec["source_port"],
                target_node=ec["target_node"],
                target_port=ec["target_port"],
            ))

        for ei in config.get("exposed_inputs", []):
            w.expose_input(ei["node_id"], ei["port_name"], ei.get("name"))

        for eo in config.get("exposed_outputs", []):
            w.expose_output(eo["node_id"], eo["port_name"], eo.get("name"))

        return w

    # --- save / load (mirrors hypergraph serialization) -------------------

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        cfg = self.to_config()
        with open(directory / "config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

        state = self.state_dict()
        if state:
            with open(directory / "checkpoint.pkl", "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        return directory

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        registry: Optional[Any] = None,
    ) -> "Workflow":
        directory = Path(directory)
        with open(directory / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        w = cls.from_config(config, registry=registry)

        ckpt_path = directory / "checkpoint.pkl"
        if ckpt_path.exists():
            with open(ckpt_path, "rb") as f:
                state = pickle.load(f)
            w.load_state_dict(state, strict=False)

        return w
