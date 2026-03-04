"""
Pipeline: graph of graphs; nodes = Graph, edges = external port connections.

Canon: WorldGenerator_2.0/TODO_04_PIPELINE.md, Pipeline_Level.md.
Contract: get_input_spec, get_output_spec, run(inputs) -> outputs (same as Graph for Stage/World).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Set, Union

from yggdrasill.foundation.graph import Graph, ValidationResult
from yggdrasill.foundation.registry import BlockRegistry

PIPELINE_CONFIG_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class PipelineEdge:
    """Edge between graph external ports: (source_node_id, source_port) -> (target_node_id, target_port)."""

    source_node: str
    source_port: str
    target_node: str
    target_port: str

    def __post_init__(self) -> None:
        for name in ("source_node", "source_port", "target_node", "target_port"):
            v = getattr(self, name)
            if not v or (isinstance(v, str) and not v.strip()):
                raise ValueError(f"{name} must be non-empty")


def _graph_port_key(entry: Dict[str, Any]) -> str:
    """Stable key for a graph port: name if set, else 'node_id:port_name' (for graph internal)."""
    if entry.get("name") is not None:
        return str(entry["name"])
    return f"{entry['node_id']}:{entry['port_name']}"


def _exposed_entry(pipeline_node_id: str, port_name: str, name: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"pipeline_node_id": pipeline_node_id, "port_name": port_name}
    if name is not None:
        out["name"] = name
    return out


class Pipeline:
    """
    Pipeline = graph of graphs. Nodes: pipeline_node_id -> Graph.
    Edges connect external outputs of one graph to external inputs of another.
    Contract: get_input_spec(), get_output_spec(), run(inputs) -> outputs (for Stage/World).
    """

    def __init__(self, pipeline_id: Optional[str] = None) -> None:
        self._pipeline_id = pipeline_id or "pipeline"
        self._nodes: Dict[str, Graph] = {}
        self._edges: List[PipelineEdge] = []
        self._in_edges_by_node: Dict[str, List[PipelineEdge]] = {}
        self._out_edges_by_node: Dict[str, List[PipelineEdge]] = {}
        self._exposed_inputs: List[Dict[str, Any]] = []
        self._exposed_outputs: List[Dict[str, Any]] = []
        self._node_trainable: Dict[str, bool] = {}
        self._metadata: Dict[str, Any] = {}
        self._execution_version = 0

    @property
    def pipeline_id(self) -> str:
        return self._pipeline_id

    @property
    def node_ids(self) -> Set[str]:
        return set(self._nodes)

    def add_graph(
        self,
        graph_or_config: Union[Graph, Dict[str, Any], str],
        pipeline_node_id: Optional[str] = None,
        *,
        registry: Optional[BlockRegistry] = None,
    ) -> str:
        """
        Add a graph as pipeline node. graph_or_config: Graph instance, config dict, or path to YAML/JSON.
        pipeline_node_id: optional; default from graph.graph_id or generated.
        """
        if isinstance(graph_or_config, Graph):
            g = graph_or_config
            nid = pipeline_node_id or g.graph_id or f"graph_{len(self._nodes)}"
        elif isinstance(graph_or_config, dict):
            reg = registry or BlockRegistry.global_registry()
            g = Graph.from_config(graph_or_config, registry=reg)
            nid = pipeline_node_id or graph_or_config.get("graph_id") or g.graph_id or f"graph_{len(self._nodes)}"
        elif isinstance(graph_or_config, str):
            g = Graph.from_yaml(graph_or_config, registry=registry or BlockRegistry.global_registry())
            nid = pipeline_node_id or g.graph_id or f"graph_{len(self._nodes)}"
        else:
            raise TypeError("graph_or_config must be Graph, dict (config), or str (path)")
        if nid in self._nodes:
            raise ValueError(f"Pipeline node already exists: {nid}")
        self._nodes[nid] = g
        self._in_edges_by_node[nid] = []
        self._out_edges_by_node[nid] = []
        self._node_trainable[nid] = True
        self._execution_version += 1
        return nid

    def add_edge(
        self,
        source_node_id: str,
        source_port: str,
        target_node_id: str,
        target_port: str,
    ) -> None:
        """Add edge between graph external ports. Ports are keys from get_output_spec/get_input_spec."""
        edge = PipelineEdge(source_node_id, source_port, target_node_id, target_port)
        self._validate_pipeline_edge(edge)
        if edge in self._edges:
            return
        self._edges.append(edge)
        self._in_edges_by_node.setdefault(target_node_id, []).append(edge)
        self._out_edges_by_node.setdefault(source_node_id, []).append(edge)
        self._execution_version += 1

    def _validate_pipeline_edge(self, edge: PipelineEdge) -> None:
        if edge.source_node not in self._nodes:
            raise ValueError(f"Source node not found: {edge.source_node}")
        if edge.target_node not in self._nodes:
            raise ValueError(f"Target node not found: {edge.target_node}")
        out_keys = {_graph_port_key(e) for e in self._nodes[edge.source_node].get_output_spec()}
        if edge.source_port not in out_keys:
            raise ValueError(
                f"Source port {edge.source_port!r} not in output spec of {edge.source_node}. "
                f"Known: {sorted(out_keys)}"
            )
        in_keys = {_graph_port_key(e) for e in self._nodes[edge.target_node].get_input_spec()}
        if edge.target_port not in in_keys:
            raise ValueError(
                f"Target port {edge.target_port!r} not in input spec of {edge.target_node}. "
                f"Known: {sorted(in_keys)}"
            )

    def expose_input(self, pipeline_node_id: str, port_name: str, name: Optional[str] = None) -> None:
        """Mark graph input as pipeline-level input."""
        if pipeline_node_id not in self._nodes:
            raise ValueError(f"Node not found: {pipeline_node_id}")
        in_keys = {_graph_port_key(e) for e in self._nodes[pipeline_node_id].get_input_spec()}
        if port_name not in in_keys:
            raise ValueError(f"Port {port_name!r} not in input spec of {pipeline_node_id}")
        self._exposed_inputs.append(_exposed_entry(pipeline_node_id, port_name, name))

    def expose_output(self, pipeline_node_id: str, port_name: str, name: Optional[str] = None) -> None:
        """Mark graph output as pipeline-level output."""
        if pipeline_node_id not in self._nodes:
            raise ValueError(f"Node not found: {pipeline_node_id}")
        out_keys = {_graph_port_key(e) for e in self._nodes[pipeline_node_id].get_output_spec()}
        if port_name not in out_keys:
            raise ValueError(f"Port {port_name!r} not in output spec of {pipeline_node_id}")
        self._exposed_outputs.append(_exposed_entry(pipeline_node_id, port_name, name))

    def _input_key(self, entry: Dict[str, Any]) -> str:
        return entry.get("name") or f"{entry['pipeline_node_id']}:{entry['port_name']}"

    def _output_key(self, entry: Dict[str, Any]) -> str:
        return entry.get("name") or f"{entry['pipeline_node_id']}:{entry['port_name']}"

    def get_input_spec(self, include_dtype: bool = False) -> List[Dict[str, Any]]:
        """Contract: list of {pipeline_node_id, port_name, name?} for run(inputs)."""
        if not include_dtype:
            return list(self._exposed_inputs)
        out = []
        for e in self._exposed_inputs:
            entry = dict(e)
            g = self._nodes.get(entry["pipeline_node_id"])
            if g is not None:
                for spec in g.get_input_spec(include_dtype=True):
                    if _graph_port_key(spec) == entry["port_name"]:
                        entry["dtype"] = spec.get("dtype")
                        break
            out.append(entry)
        return out

    def get_output_spec(self, include_dtype: bool = False) -> List[Dict[str, Any]]:
        """Contract: list of {pipeline_node_id, port_name, name?} for run() return keys."""
        if not include_dtype:
            return list(self._exposed_outputs)
        out = []
        for e in self._exposed_outputs:
            entry = dict(e)
            g = self._nodes.get(entry["pipeline_node_id"])
            if g is not None:
                for spec in g.get_output_spec(include_dtype=True):
                    if _graph_port_key(spec) == entry["port_name"]:
                        entry["dtype"] = spec.get("dtype")
                        break
            out.append(entry)
        return out

    def get_graph(self, pipeline_node_id: str) -> Optional[Graph]:
        return self._nodes.get(pipeline_node_id)

    def get_edges(self) -> List[PipelineEdge]:
        return list(self._edges)

    def get_edges_in(self, pipeline_node_id: str) -> List[PipelineEdge]:
        return list(self._in_edges_by_node.get(pipeline_node_id, []))

    def get_edges_out(self, pipeline_node_id: str) -> List[PipelineEdge]:
        return list(self._out_edges_by_node.get(pipeline_node_id, []))

    def validate(self, strict: bool = True, validate_graphs: bool = False) -> ValidationResult:
        """Validate: DAG, reachability, port compatibility; optionally run graph.validate() for each node."""
        errors: List[str] = []
        warnings: List[str] = []
        # DAG check via topological sort
        in_deg = {nid: 0 for nid in self._nodes}
        for e in self._edges:
            in_deg[e.target_node] = in_deg.get(e.target_node, 0) + 1
        queue = [nid for nid, d in in_deg.items() if d == 0]
        ordered = []
        while queue:
            nid = queue.pop(0)
            ordered.append(nid)
            for e in self.get_edges_out(nid):
                in_deg[e.target_node] -= 1
                if in_deg[e.target_node] == 0:
                    queue.append(e.target_node)
        if len(ordered) != len(self._nodes):
            errors.append("Pipeline graph has a cycle (not a DAG)")
        # Reachability
        if self._nodes and (self._exposed_inputs or self._exposed_outputs):
            reachable = self._reachable_from_inputs()
            reaches_out = self._reaches_outputs()
            for nid in self._nodes:
                if nid not in reachable and self._exposed_inputs:
                    warnings.append(f"Node {nid} is not reachable from any exposed input")
                if nid not in reaches_out and self._exposed_outputs:
                    warnings.append(f"Node {nid} does not lead to any exposed output")
        # Optional: validate each graph
        if validate_graphs:
            for nid, g in self._nodes.items():
                try:
                    g.validate(strict=True)
                except ValueError as err:
                    errors.append(f"Graph {nid}: {err}")
        result = ValidationResult(errors=errors, warnings=warnings)
        if strict and result.errors:
            raise ValueError("; ".join(result.errors))
        return result

    def _reachable_from_inputs(self) -> Set[str]:
        start = {e["pipeline_node_id"] for e in self._exposed_inputs}
        reachable = set(start)
        queue = list(start)
        while queue:
            nid = queue.pop(0)
            for e in self.get_edges_out(nid):
                if e.target_node not in reachable:
                    reachable.add(e.target_node)
                    queue.append(e.target_node)
        return reachable

    def _reaches_outputs(self) -> Set[str]:
        start = {e["pipeline_node_id"] for e in self._exposed_outputs}
        reaches = set(start)
        queue = list(start)
        while queue:
            nid = queue.pop(0)
            for e in self.get_edges_in(nid):
                if e.source_node not in reaches:
                    reaches.add(e.source_node)
                    queue.append(e.source_node)
        return reaches

    def infer_exposed_ports(self) -> None:
        """Set exposed I/O by inference: input = graph port with no incoming pipeline edge; output = no outgoing."""
        self._exposed_inputs = []
        self._exposed_outputs = []
        in_ports_covered = {(e.target_node, e.target_port) for e in self._edges}
        out_ports_covered = {(e.source_node, e.source_port) for e in self._edges}
        for nid, g in self._nodes.items():
            for spec in g.get_input_spec():
                key = _graph_port_key(spec)
                if (nid, key) not in in_ports_covered:
                    self._exposed_inputs.append(_exposed_entry(nid, key, None))
            for spec in g.get_output_spec():
                key = _graph_port_key(spec)
                if (nid, key) not in out_ports_covered:
                    self._exposed_outputs.append(_exposed_entry(nid, key, None))

    def trainable_parameters(self) -> Iterator[Any]:
        """Parameters from trainable graphs only."""
        for nid, g in self._nodes.items():
            if self._node_trainable.get(nid, True):
                yield from g.trainable_parameters()

    def set_trainable(self, pipeline_node_id: str, trainable: bool) -> None:
        if pipeline_node_id not in self._nodes:
            raise ValueError(f"Node not found: {pipeline_node_id}")
        self._node_trainable[pipeline_node_id] = trainable

    def to(self, device: Any) -> "Pipeline":
        for g in self._nodes.values():
            g.to(device)
        return self

    def run(
        self,
        inputs: Dict[str, Any],
        *,
        training: bool = False,
        device: Any = None,
        callbacks: Optional[List[Any]] = None,
        **graph_run_kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute pipeline; delegates to pipeline executor."""
        from yggdrasill.pipeline.run import run_pipeline
        return run_pipeline(
            self, inputs,
            training=training, device=device, callbacks=callbacks,
            **graph_run_kwargs,
        )

    def to_config(self) -> Dict[str, Any]:
        """Serialize structure (graphs as configs, edges, exposed I/O)."""
        graphs_cfg = []
        for nid, g in self._nodes.items():
            graphs_cfg.append({
                "pipeline_node_id": nid,
                "graph_config": g.to_config(),
                "trainable": self._node_trainable.get(nid, True),
            })
        edges_cfg = [
            {
                "source_node": e.source_node,
                "source_port": e.source_port,
                "target_node": e.target_node,
                "target_port": e.target_port,
            }
            for e in self._edges
        ]
        out: Dict[str, Any] = {
            "schema_version": PIPELINE_CONFIG_SCHEMA_VERSION,
            "pipeline_id": self._pipeline_id,
            "graphs": graphs_cfg,
            "edges": edges_cfg,
            "exposed_inputs": list(self._exposed_inputs),
            "exposed_outputs": list(self._exposed_outputs),
        }
        if self._metadata:
            out["metadata"] = dict(self._metadata)
        return out

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        registry: Optional[BlockRegistry] = None,
        validate: bool = False,
    ) -> "Pipeline":
        """Build pipeline from config."""
        reg = registry or BlockRegistry.global_registry()
        p = cls(pipeline_id=config.get("pipeline_id", "pipeline"))
        p._metadata = dict(config.get("metadata", {}))
        for gc in config.get("graphs", []):
            nid = gc["pipeline_node_id"]
            graph_cfg = gc.get("graph_config")
            if isinstance(graph_cfg, dict) and set(graph_cfg) == {"ref"}:
                path = graph_cfg["ref"]
                try:
                    from omegaconf import OmegaConf
                    graph_cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
                except Exception:
                    with open(path, "r", encoding="utf-8") as f:
                        graph_cfg = json.load(f)
            g = Graph.from_config(graph_cfg or {}, registry=reg)
            p._nodes[nid] = g
            p._in_edges_by_node[nid] = []
            p._out_edges_by_node[nid] = []
            p._node_trainable[nid] = gc.get("trainable", True)
        for ec in config.get("edges", []):
            edge = PipelineEdge(
                source_node=ec["source_node"],
                source_port=ec["source_port"],
                target_node=ec["target_node"],
                target_port=ec["target_port"],
            )
            p._edges.append(edge)
            p._in_edges_by_node.setdefault(edge.target_node, []).append(edge)
            p._out_edges_by_node.setdefault(edge.source_node, []).append(edge)
        for e in config.get("exposed_inputs", []):
            p._exposed_inputs.append(_exposed_entry(
                e["pipeline_node_id"], e["port_name"], e.get("name")
            ))
        for e in config.get("exposed_outputs", []):
            p._exposed_outputs.append(_exposed_entry(
                e["pipeline_node_id"], e["port_name"], e.get("name")
            ))
        if validate:
            p.validate(strict=True)
        return p

    @classmethod
    def from_yaml(cls, path: str, *, registry: Optional[BlockRegistry] = None, validate: bool = False) -> "Pipeline":
        """
        Load pipeline config from a YAML or JSON file and build via from_config.
        Same contract as Graph.from_yaml; uses OmegaConf if available for YAML, else JSON.
        """
        try:
            from omegaconf import OmegaConf
            config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        except Exception:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
        return cls.from_config(config, registry=registry, validate=validate)

    def state_dict(self) -> Dict[str, Any]:
        """Aggregated state_dict of all graphs keyed by pipeline_node_id."""
        out: Dict[str, Any] = {}
        for nid, g in self._nodes.items():
            sd = g.state_dict()
            if sd:
                out[nid] = sd
        return out

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        for nid, g in self._nodes.items():
            if nid in state:
                g.load_state_dict(state[nid], strict=strict)
        if strict:
            for key in state:
                if key not in self._nodes:
                    raise KeyError(f"Unknown pipeline node in checkpoint: {key}")

    def save_config(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_config(), f, indent=2)

    def save_checkpoint(self, path: str, format: str = "dir", backend: str = "json") -> None:
        """Save checkpoint: directory with subdir per pipeline_node_id (graph checkpoint)."""
        if backend != "json":
            raise NotImplementedError(f"Checkpoint backend {backend!r} not implemented for pipeline")
        os.makedirs(path, exist_ok=True)
        for nid, g in self._nodes.items():
            sub = os.path.join(path, nid)
            os.makedirs(sub, exist_ok=True)
            g.save_config(os.path.join(sub, "config.json"))
            g.save_checkpoint(os.path.join(sub, "checkpoint.json"), format="single", backend="json")
        meta = {"pipeline_id": self._pipeline_id, "node_ids": list(self._nodes)}
        with open(os.path.join(path, "pipeline_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load_from_checkpoint(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        registry: Optional[BlockRegistry] = None,
        backend: str = "json",
    ) -> None:
        """Load: optionally rebuild from config, then load graph checkpoints from checkpoint_dir."""
        if backend != "json":
            raise NotImplementedError(f"Checkpoint backend {backend!r} not implemented for pipeline")
        if config_path is not None:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        if config is not None:
            new_p = Pipeline.from_config(config, registry=registry)
            self._pipeline_id = new_p._pipeline_id
            self._nodes = new_p._nodes
            self._edges = new_p._edges
            self._in_edges_by_node = new_p._in_edges_by_node
            self._out_edges_by_node = new_p._out_edges_by_node
            self._exposed_inputs = new_p._exposed_inputs
            self._exposed_outputs = new_p._exposed_outputs
            self._node_trainable = new_p._node_trainable
        if checkpoint_dir is not None:
            for nid in self._nodes:
                sub = os.path.join(checkpoint_dir, nid)
                ckpt = os.path.join(sub, "checkpoint.json")
                if os.path.isfile(ckpt):
                    self._nodes[nid].load_from_checkpoint(checkpoint_path=ckpt)

    def save(self, save_dir: str) -> None:
        """Save config and checkpoint to directory."""
        os.makedirs(save_dir, exist_ok=True)
        self.save_config(os.path.join(save_dir, "config.json"))
        self.save_checkpoint(os.path.join(save_dir, "checkpoints"))

    @classmethod
    def load(cls, save_dir: str, *, registry: Optional[BlockRegistry] = None) -> "Pipeline":
        """Load pipeline from directory (config + checkpoints)."""
        config_path = os.path.join(save_dir, "config.json")
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"No config at {config_path}")
        p = cls.from_config(json.load(open(config_path, "r", encoding="utf-8")), registry=registry)
        if os.path.isdir(checkpoint_dir):
            p.load_from_checkpoint(checkpoint_dir=checkpoint_dir, registry=registry)
        return p
