"""Workflow: a hypergraph of hypergraphs.

Nodes are Hypergraph instances.  Edges connect their exposed outputs
to other hypergraphs' exposed inputs.  The same engine (Validator,
Planner, Executor) runs the workflow without any changes.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

from yggdrasill.engine.edge import Edge
from yggdrasill.engine.structure import Hypergraph


class Workflow:
    """Hypergraph of hypergraphs -- implements the same structural protocol
    that the engine expects (node_ids, get_node, get_edges, etc.)."""

    def __init__(self, workflow_id: Optional[str] = None) -> None:
        self._workflow_id = workflow_id or "workflow"
        self._workflow_kind: Optional[str] = None
        self._metadata: Dict[str, Any] = {}

        self._nodes: Dict[str, Hypergraph] = {}
        self._edges: List[Edge] = []
        self._in_edges: Dict[str, List[Edge]] = {}
        self._out_edges: Dict[str, List[Edge]] = {}

        self._exposed_inputs: List[Dict[str, Any]] = []
        self._exposed_outputs: List[Dict[str, Any]] = []

        self._execution_version: int = 0
        self._node_trainable: Dict[str, bool] = {}

    # --- identity / metadata -----------------------------------------------

    @property
    def workflow_id(self) -> str:
        return self._workflow_id

    @property
    def graph_id(self) -> str:
        return self._workflow_id

    @property
    def workflow_kind(self) -> Optional[str]:
        return self._workflow_kind

    @workflow_kind.setter
    def workflow_kind(self, value: Optional[str]) -> None:
        self._workflow_kind = value

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        self._metadata = dict(value)

    @property
    def execution_version(self) -> int:
        return self._execution_version

    # --- structural protocol (engine-facing) --------------------------------

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
        result: List[Dict[str, Any]] = []
        for entry in self._exposed_inputs:
            rec: Dict[str, Any] = dict(entry)
            if include_dtype:
                graph = self._nodes.get(entry["node_id"])
                if graph is not None:
                    for sp in graph.get_input_spec(include_dtype=True):
                        if sp["port_name"] == entry["port_name"]:
                            if "dtype" in sp:
                                rec["dtype"] = sp["dtype"]
                            break
            result.append(rec)
        return result

    def get_output_spec(self, include_dtype: bool = False) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for entry in self._exposed_outputs:
            rec: Dict[str, Any] = dict(entry)
            if include_dtype:
                graph = self._nodes.get(entry["node_id"])
                if graph is not None:
                    for sp in graph.get_output_spec(include_dtype=True):
                        if sp["port_name"] == entry["port_name"]:
                            if "dtype" in sp:
                                rec["dtype"] = sp["dtype"]
                            break
            result.append(rec)
        return result

    # --- node management ----------------------------------------------------

    def add_node(self, graph_id: str, hypergraph: Hypergraph) -> str:
        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id must be non-empty")
        graph_id = graph_id.strip()
        self._nodes[graph_id] = hypergraph
        self._in_edges.setdefault(graph_id, [])
        self._out_edges.setdefault(graph_id, [])
        self._node_trainable.setdefault(graph_id, True)
        self._execution_version += 1
        return graph_id

    def remove_node(self, graph_id: str) -> None:
        if graph_id not in self._nodes:
            return
        del self._nodes[graph_id]
        self._edges = [
            e for e in self._edges
            if e.source_node != graph_id and e.target_node != graph_id
        ]
        self._in_edges.pop(graph_id, None)
        self._out_edges.pop(graph_id, None)
        for nid in list(self._in_edges):
            self._in_edges[nid] = [
                e for e in self._in_edges[nid] if e.source_node != graph_id
            ]
        for nid in list(self._out_edges):
            self._out_edges[nid] = [
                e for e in self._out_edges[nid] if e.target_node != graph_id
            ]
        self._exposed_inputs = [
            ei for ei in self._exposed_inputs if ei.get("node_id") != graph_id
        ]
        self._exposed_outputs = [
            eo for eo in self._exposed_outputs if eo.get("node_id") != graph_id
        ]
        self._node_trainable.pop(graph_id, None)
        self._execution_version += 1

    # --- edge management ----------------------------------------------------

    def add_edge(
        self,
        source_graph_id: str,
        source_port: str,
        target_graph_id: str,
        target_port: str,
    ) -> None:
        if source_graph_id not in self._nodes:
            raise ValueError(f"Source graph '{source_graph_id}' not in workflow")
        if target_graph_id not in self._nodes:
            raise ValueError(f"Target graph '{target_graph_id}' not in workflow")

        src_graph = self._nodes[source_graph_id]
        dst_graph = self._nodes[target_graph_id]

        src_port_names = {
            sp["port_name"] for sp in src_graph.get_output_spec()
        }
        if source_port not in src_port_names:
            raise ValueError(
                f"Port '{source_port}' not in output spec of graph '{source_graph_id}'"
            )

        dst_port_names = {
            sp["port_name"] for sp in dst_graph.get_input_spec()
        }
        if target_port not in dst_port_names:
            raise ValueError(
                f"Port '{target_port}' not in input spec of graph '{target_graph_id}'"
            )

        edge = Edge(
            source_node=source_graph_id,
            source_port=source_port,
            target_node=target_graph_id,
            target_port=target_port,
        )
        if edge in self._edges:
            return
        self._edges.append(edge)
        self._in_edges.setdefault(edge.target_node, []).append(edge)
        self._out_edges.setdefault(edge.source_node, []).append(edge)
        self._execution_version += 1

    def remove_edge(
        self,
        source_graph_id: str,
        source_port: str,
        target_graph_id: str,
        target_port: str,
    ) -> None:
        edge = Edge(
            source_node=source_graph_id,
            source_port=source_port,
            target_node=target_graph_id,
            target_port=target_port,
        )
        if edge not in self._edges:
            return
        self._edges.remove(edge)
        in_list = self._in_edges.get(edge.target_node, [])
        if edge in in_list:
            in_list.remove(edge)
        out_list = self._out_edges.get(edge.source_node, [])
        if edge in out_list:
            out_list.remove(edge)
        self._execution_version += 1

    # --- exposed ports ------------------------------------------------------

    def expose_input(
        self, graph_id: str, port_name: str, name: Optional[str] = None,
    ) -> None:
        if graph_id not in self._nodes:
            raise ValueError(f"Graph '{graph_id}' not in workflow")
        graph = self._nodes[graph_id]
        input_port_names = {sp["port_name"] for sp in graph.get_input_spec()}
        if port_name not in input_port_names:
            raise ValueError(
                f"Port '{port_name}' not in input spec of graph '{graph_id}'"
            )
        entry: Dict[str, Any] = {"node_id": graph_id, "port_name": port_name}
        if name is not None:
            entry["name"] = name
        for existing in self._exposed_inputs:
            if existing.get("node_id") == graph_id and existing.get("port_name") == port_name:
                return
        self._exposed_inputs.append(entry)
        self._execution_version += 1

    def expose_output(
        self, graph_id: str, port_name: str, name: Optional[str] = None,
    ) -> None:
        if graph_id not in self._nodes:
            raise ValueError(f"Graph '{graph_id}' not in workflow")
        graph = self._nodes[graph_id]
        output_port_names = {sp["port_name"] for sp in graph.get_output_spec()}
        if port_name not in output_port_names:
            raise ValueError(
                f"Port '{port_name}' not in output spec of graph '{graph_id}'"
            )
        entry: Dict[str, Any] = {"node_id": graph_id, "port_name": port_name}
        if name is not None:
            entry["name"] = name
        for existing in self._exposed_outputs:
            if existing.get("node_id") == graph_id and existing.get("port_name") == port_name:
                return
        self._exposed_outputs.append(entry)
        self._execution_version += 1

    # --- infer exposed ports ------------------------------------------------

    def infer_exposed_ports(self) -> None:
        """Auto-detect exposed inputs/outputs by absence of workflow edges."""
        self._exposed_inputs.clear()
        self._exposed_outputs.clear()

        for graph_id, graph in self._nodes.items():
            in_edges = self.get_edges_in(graph_id)
            covered_in = {e.target_port for e in in_edges}
            for sp in graph.get_input_spec():
                if sp["port_name"] not in covered_in:
                    self._exposed_inputs.append({
                        "node_id": graph_id,
                        "port_name": sp["port_name"],
                    })

            out_edges = self.get_edges_out(graph_id)
            covered_out = {e.source_port for e in out_edges}
            for sp in graph.get_output_spec():
                if sp["port_name"] not in covered_out:
                    self._exposed_outputs.append({
                        "node_id": graph_id,
                        "port_name": sp["port_name"],
                    })

        self._execution_version += 1

    # --- run (delegates to engine) ------------------------------------------

    def run(
        self,
        inputs: Dict[str, Any],
        *,
        training: bool = False,
        num_loop_steps: Optional[int] = None,
        device: Optional[Any] = None,
        callbacks: Optional[List[Callable[..., None]]] = None,
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

    # --- state dict ---------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for nid, hg in self._nodes.items():
            result[nid] = hg.state_dict()
        return result

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        if strict:
            extra = set(state.keys()) - set(self._nodes.keys())
            if extra:
                raise KeyError(f"state_dict has keys not in workflow: {extra}")
        for nid, hg in self._nodes.items():
            if nid in state:
                hg.load_state_dict(state[nid], strict=strict)

    # --- device / trainable -------------------------------------------------

    def to(self, device: Any) -> "Workflow":
        for hg in self._nodes.values():
            if hasattr(hg, "to") and callable(hg.to):
                hg.to(device)
        return self

    def set_trainable(self, graph_id: str, trainable: bool) -> None:
        if graph_id not in self._nodes:
            raise ValueError(f"Graph '{graph_id}' not in workflow")
        self._node_trainable[graph_id] = trainable

    def trainable_parameters(self) -> Iterator[Any]:
        for gid, hg in self._nodes.items():
            if not self._node_trainable.get(gid, True):
                continue
            if hasattr(hg, "trainable_parameters"):
                yield from hg.trainable_parameters()

    # --- config -------------------------------------------------------------

    def to_config(self) -> Dict[str, Any]:
        graphs = []
        for gid, hg in self._nodes.items():
            entry: Dict[str, Any] = {
                "graph_id": gid,
                "config": hg.to_config(),
            }
            if not self._node_trainable.get(gid, True):
                entry["trainable"] = False
            graphs.append(entry)

        edges = [
            {
                "source_graph": e.source_node,
                "source_port": e.source_port,
                "target_graph": e.target_node,
                "target_port": e.target_port,
            }
            for e in self._edges
        ]

        cfg: Dict[str, Any] = {
            "schema_version": "1.0",
            "workflow_id": self._workflow_id,
            "graphs": graphs,
            "edges": edges,
            "exposed_inputs": [dict(ei) for ei in self._exposed_inputs],
            "exposed_outputs": [dict(eo) for eo in self._exposed_outputs],
        }
        if self._workflow_kind is not None:
            cfg["workflow_kind"] = self._workflow_kind
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
        w.workflow_kind = config.get("workflow_kind")
        w.metadata = dict(config.get("metadata", {}))

        for gc in config.get("graphs", []):
            gid = gc["graph_id"]
            hg = Hypergraph.from_config(gc["config"], registry=registry)
            w.add_node(gid, hg)
            if "trainable" in gc:
                w._node_trainable[gid] = gc["trainable"]

        for ec in config.get("edges", []):
            w.add_edge(
                ec["source_graph"],
                ec["source_port"],
                ec["target_graph"],
                ec["target_port"],
            )

        for ei in config.get("exposed_inputs", []):
            w.expose_input(
                ei.get("node_id") or ei.get("graph_id"),
                ei["port_name"],
                ei.get("name"),
            )

        for eo in config.get("exposed_outputs", []):
            w.expose_output(
                eo.get("node_id") or eo.get("graph_id"),
                eo["port_name"],
                eo.get("name"),
            )

        return w

    # --- save / load --------------------------------------------------------

    def save(
        self,
        directory: str | Path,
        *,
        config_filename: str = "config.json",
        checkpoint_filename: str = "checkpoint.pkl",
    ) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        cfg = self.to_config()
        with open(directory / config_filename, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

        state = self.state_dict()
        with open(directory / checkpoint_filename, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        return directory

    def save_config(
        self,
        directory: str | Path,
        *,
        filename: str = "config.json",
    ) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        cfg = self.to_config()
        with open(directory / filename, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        return directory

    def save_checkpoint(
        self,
        directory: str | Path,
        *,
        filename: str = "checkpoint.pkl",
    ) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        state = self.state_dict()
        with open(directory / filename, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        return directory

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        registry: Optional[Any] = None,
        config_filename: str = "config.json",
        checkpoint_filename: str = "checkpoint.pkl",
        load_checkpoint_flag: bool = True,
    ) -> "Workflow":
        directory = Path(directory)
        with open(directory / config_filename, "r", encoding="utf-8") as f:
            config = json.load(f)

        w = cls.from_config(config, registry=registry)

        if load_checkpoint_flag:
            ckpt_path = directory / checkpoint_filename
            if ckpt_path.exists():
                with open(ckpt_path, "rb") as f:
                    state = pickle.load(f)
                w.load_state_dict(state, strict=False)

        return w

    def load_from_checkpoint(
        self,
        directory: str | Path,
        *,
        checkpoint_filename: str = "checkpoint.pkl",
    ) -> None:
        directory = Path(directory)
        ckpt_path = directory / checkpoint_filename
        with open(ckpt_path, "rb") as f:
            state = pickle.load(f)
        self.load_state_dict(state, strict=False)
