from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

from yggdrasill.engine.edge import Edge
from yggdrasill.foundation.node import AbstractGraphNode
from yggdrasill.foundation.port import PortDirection

_instance_counter = itertools.count()


def _resolve_config_ref(config: Dict[str, Any]) -> Dict[str, Any]:
    """If *config* is ``{"ref": "path/to/file"}`` load from that file."""
    if set(config.keys()) == {"ref"}:
        ref_path = Path(config["ref"])
        if ref_path.suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]
                with open(ref_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except ImportError:
                raise ImportError(
                    f"PyYAML required to load YAML config ref: {ref_path}"
                )
        if ref_path.suffix not in (".json",):
            raise ValueError(
                f"Unsupported config ref file extension '{ref_path.suffix}'; "
                f"expected .json, .yaml, or .yml"
            )
        with open(ref_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return config


class Hypergraph:
    """Stores nodes, edges, and exposed inputs/outputs.

    Serves as the structural backbone that the engine (validator, planner,
    executor) operates on.  At the task-hypergraph level, nodes are task-node
    objects (Block+Node).  At the workflow level a separate Workflow class
    wraps the same protocol with Hypergraph instances as "nodes".
    """

    def __init__(self, graph_id: Optional[str] = None) -> None:
        self._instance_id = next(_instance_counter)
        self._graph_id = graph_id or "graph"
        self._graph_kind: Optional[str] = None
        self._metadata: Dict[str, Any] = {}

        self._nodes: Dict[str, Any] = {}
        self._edges: List[Edge] = []
        self._in_edges: Dict[str, List[Edge]] = {}
        self._out_edges: Dict[str, List[Edge]] = {}

        self._exposed_inputs: List[Dict[str, Any]] = []
        self._exposed_outputs: List[Dict[str, Any]] = []

        self._execution_version: int = 0
        self._node_trainable: Dict[str, bool] = {}

    # --- identity --------------------------------------------------------

    @property
    def graph_id(self) -> str:
        return self._graph_id

    @property
    def graph_kind(self) -> Optional[str]:
        return self._graph_kind

    @graph_kind.setter
    def graph_kind(self, value: Optional[str]) -> None:
        self._graph_kind = value

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        self._metadata = dict(value)

    @property
    def execution_version(self) -> int:
        return self._execution_version

    # --- nodes -----------------------------------------------------------

    @property
    def node_ids(self) -> Set[str]:
        return set(self._nodes.keys())

    def get_node(self, node_id: str) -> Optional[Any]:
        return self._nodes.get(node_id)

    def add_node(self, node_id: str, node: Any) -> None:
        """Add a ready-made node (task-node object) to the graph."""
        if not node_id or not node_id.strip():
            raise ValueError("node_id must be non-empty")
        node_id = node_id.strip()
        self._nodes[node_id] = node
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
            ei for ei in self._exposed_inputs if ei["node_id"] != node_id
        ]
        self._exposed_outputs = [
            eo for eo in self._exposed_outputs if eo["node_id"] != node_id
        ]
        self._node_trainable.pop(node_id, None)
        self._execution_version += 1

    # --- edges -----------------------------------------------------------

    def add_edge(self, edge: Edge) -> None:
        if edge.source_node not in self._nodes:
            raise ValueError(f"Source node '{edge.source_node}' not in graph")
        if edge.target_node not in self._nodes:
            raise ValueError(f"Target node '{edge.target_node}' not in graph")

        src_node = self._nodes[edge.source_node]
        dst_node = self._nodes[edge.target_node]

        src_port = None
        dst_port = None

        if isinstance(src_node, AbstractGraphNode):
            src_port = src_node.get_port(edge.source_port)
            if src_port is None:
                raise ValueError(
                    f"Port '{edge.source_port}' not found on node '{edge.source_node}'"
                )
            if src_port.direction != PortDirection.OUT:
                raise ValueError(
                    f"Port '{edge.source_port}' on '{edge.source_node}' is not an output"
                )

        if isinstance(dst_node, AbstractGraphNode):
            dst_port = dst_node.get_port(edge.target_port)
            if dst_port is None:
                raise ValueError(
                    f"Port '{edge.target_port}' not found on node '{edge.target_node}'"
                )
            if dst_port.direction != PortDirection.IN:
                raise ValueError(
                    f"Port '{edge.target_port}' on '{edge.target_node}' is not an input"
                )

        if isinstance(src_node, AbstractGraphNode) and isinstance(dst_node, AbstractGraphNode):
            if src_port is not None and dst_port is not None:
                if not src_port.compatible_with(dst_port):
                    raise ValueError(
                        f"Incompatible port types: {edge.source_node}.{edge.source_port} "
                        f"({src_port.dtype}) -> {edge.target_node}.{edge.target_port} "
                        f"({dst_port.dtype})"
                    )

        if edge in self._edges:
            return  # idempotent

        self._edges.append(edge)
        self._in_edges.setdefault(edge.target_node, []).append(edge)
        self._out_edges.setdefault(edge.source_node, []).append(edge)
        self._execution_version += 1

    def remove_edge(self, edge: Edge) -> None:
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

    def get_edges(self) -> List[Edge]:
        return list(self._edges)

    def get_edges_in(self, node_id: str) -> List[Edge]:
        return list(self._in_edges.get(node_id, []))

    def get_edges_out(self, node_id: str) -> List[Edge]:
        return list(self._out_edges.get(node_id, []))

    # --- exposed inputs / outputs ----------------------------------------

    def expose_input(
        self, node_id: str, port_name: str, name: Optional[str] = None,
    ) -> None:
        if node_id not in self._nodes:
            raise ValueError(f"Node '{node_id}' not in graph")
        node = self._nodes[node_id]
        if isinstance(node, AbstractGraphNode):
            port = node.get_port(port_name)
            if port is None:
                raise ValueError(f"Port '{port_name}' not found on node '{node_id}'")
            if port.direction != PortDirection.IN:
                raise ValueError(f"Port '{port_name}' on node '{node_id}' is not an input")
        entry: Dict[str, Any] = {"node_id": node_id, "port_name": port_name}
        if name is not None:
            entry["name"] = name
        for existing in self._exposed_inputs:
            if existing["node_id"] == node_id and existing["port_name"] == port_name:
                return
        self._exposed_inputs.append(entry)
        self._execution_version += 1

    def expose_output(
        self, node_id: str, port_name: str, name: Optional[str] = None,
    ) -> None:
        if node_id not in self._nodes:
            raise ValueError(f"Node '{node_id}' not in graph")
        node = self._nodes[node_id]
        if isinstance(node, AbstractGraphNode):
            port = node.get_port(port_name)
            if port is None:
                raise ValueError(f"Port '{port_name}' not found on node '{node_id}'")
            if port.direction != PortDirection.OUT:
                raise ValueError(f"Port '{port_name}' on node '{node_id}' is not an output")
        entry: Dict[str, Any] = {"node_id": node_id, "port_name": port_name}
        if name is not None:
            entry["name"] = name
        for existing in self._exposed_outputs:
            if existing["node_id"] == node_id and existing["port_name"] == port_name:
                return
        self._exposed_outputs.append(entry)
        self._execution_version += 1

    def get_input_spec(self, include_dtype: bool = False) -> List[Dict[str, Any]]:
        result = []
        for entry in self._exposed_inputs:
            rec: Dict[str, Any] = dict(entry)
            if include_dtype:
                node = self._nodes.get(entry["node_id"])
                if node is not None and isinstance(node, AbstractGraphNode):
                    port = node.get_port(entry["port_name"])
                    if port is not None:
                        rec["dtype"] = port.dtype.value if hasattr(port.dtype, "value") else str(port.dtype)
            result.append(rec)
        return result

    def get_output_spec(self, include_dtype: bool = False) -> List[Dict[str, Any]]:
        result = []
        for entry in self._exposed_outputs:
            rec: Dict[str, Any] = dict(entry)
            if include_dtype:
                node = self._nodes.get(entry["node_id"])
                if node is not None and isinstance(node, AbstractGraphNode):
                    port = node.get_port(entry["port_name"])
                    if port is not None:
                        rec["dtype"] = port.dtype.value if hasattr(port.dtype, "value") else str(port.dtype)
            result.append(rec)
        return result

    # --- Phase 3: config-driven API --------------------------------------

    def add_node_from_config(
        self,
        node_id: str,
        block_type: str,
        *,
        config: Optional[Dict[str, Any]] = None,
        block_id: Optional[str] = None,
        pretrained: Optional[Any] = None,
        trainable: bool = True,
        registry: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Create a task-node via the registry and add it to the graph."""
        from yggdrasill.foundation.registry import BlockRegistry

        if not node_id or not node_id.strip():
            raise ValueError("node_id must be non-empty")
        node_id = node_id.strip()
        if node_id in self._nodes:
            raise ValueError(f"node_id '{node_id}' already exists in graph")

        reg = registry or BlockRegistry.global_registry()
        build_cfg: Dict[str, Any] = {"block_type": block_type, "node_id": node_id}
        if block_id is not None:
            build_cfg["block_id"] = block_id
        if config:
            safe_cfg = {
                k: v for k, v in config.items()
                if k not in ("block_type", "node_id", "block_id")
            }
            build_cfg.update(safe_cfg)

        node = reg.build(build_cfg)
        self.add_node(node_id, node)
        self._node_trainable[node_id] = trainable

        if pretrained is not None:
            if isinstance(pretrained, (str, Path)):
                from yggdrasill.hypergraph.serialization import _read_checkpoint
                pretrained = _read_checkpoint(Path(pretrained))
            if isinstance(pretrained, dict):
                node.load_state_dict(pretrained, strict=False)

        if kwargs.get("auto_connect") and hasattr(self, "_auto_connect_fn"):
            self._auto_connect_fn(self, node_id, node)

        return node_id

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        *,
        registry: Optional[Any] = None,
        validate: bool = False,
    ) -> "Hypergraph":
        """Build a Hypergraph from a config dict."""
        import warnings

        from yggdrasill.foundation.registry import BlockRegistry

        sv = config.get("schema_version")
        if sv is not None and sv != "1.0":
            warnings.warn(
                f"Config schema_version '{sv}' differs from supported '1.0'; "
                f"loading may produce unexpected results",
                stacklevel=2,
            )

        reg = registry or BlockRegistry.global_registry()
        g = cls(graph_id=config.get("graph_id", "graph"))
        g.graph_kind = config.get("graph_kind")
        g.metadata = dict(config.get("metadata", {}))

        for nc in config.get("nodes", []):
            nid = nc["node_id"]
            bt = nc["block_type"]
            node_cfg = _resolve_config_ref(nc.get("config") or {})
            bid = nc.get("block_id")
            build_cfg: Dict[str, Any] = {"block_type": bt, "node_id": nid}
            if bid is not None:
                build_cfg["block_id"] = bid
            if node_cfg:
                safe_cfg = {
                    k: v for k, v in node_cfg.items()
                    if k not in ("block_type", "node_id", "block_id")
                }
                build_cfg.update(safe_cfg)
            node = reg.build(build_cfg)
            g.add_node(nid, node)
            g._node_trainable[nid] = nc.get("trainable", True)

        for ec in config.get("edges", []):
            g.add_edge(Edge(
                source_node=ec["source_node"],
                source_port=ec["source_port"],
                target_node=ec["target_node"],
                target_port=ec["target_port"],
            ))

        for ei in config.get("exposed_inputs", []):
            g.expose_input(ei["node_id"], ei["port_name"], ei.get("name"))

        for eo in config.get("exposed_outputs", []):
            g.expose_output(eo["node_id"], eo["port_name"], eo.get("name"))

        if validate:
            from yggdrasill.engine.validator import validate as _validate
            result = _validate(g)
            if not result.valid:
                raise ValueError(f"Config validation failed: {result.errors}")

        return g

    def to_config(self) -> Dict[str, Any]:
        """Export the structure as a JSON-serialisable dict (no weights)."""
        nodes = []
        for nid, node in self._nodes.items():
            entry: Dict[str, Any] = {"node_id": nid}
            if hasattr(node, "block_type"):
                entry["block_type"] = node.block_type
            if hasattr(node, "block_id"):
                entry["block_id"] = node.block_id
            if hasattr(node, "config"):
                entry["config"] = node.config
            entry["trainable"] = self._node_trainable.get(nid, True)
            nodes.append(entry)

        edges = [
            {
                "source_node": e.source_node,
                "source_port": e.source_port,
                "target_node": e.target_node,
                "target_port": e.target_port,
            }
            for e in self._edges
        ]

        result: Dict[str, Any] = {
            "schema_version": "1.0",
            "graph_id": self._graph_id,
            "nodes": nodes,
            "edges": edges,
            "exposed_inputs": [dict(ei) for ei in self._exposed_inputs],
            "exposed_outputs": [dict(eo) for eo in self._exposed_outputs],
        }
        if self._graph_kind is not None:
            result["graph_kind"] = self._graph_kind
        if self._metadata:
            result["metadata"] = dict(self._metadata)
        return result

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
        """Execute the hypergraph; delegates to engine.run(self, ...)."""
        from yggdrasill.engine.executor import run as _run
        return _run(
            self, inputs,
            training=training,
            num_loop_steps=num_loop_steps,
            device=device,
            callbacks=callbacks,
            dry_run=dry_run,
            validate_before=validate_before,
            **kwargs,
        )

    def infer_exposed_ports(self) -> None:
        """Auto-detect exposed inputs/outputs from uncovered ports."""
        self._exposed_inputs.clear()
        self._exposed_outputs.clear()
        for nid, node in self._nodes.items():
            if not isinstance(node, AbstractGraphNode):
                continue
            in_edges = self.get_edges_in(nid)
            covered_in = {e.target_port for e in in_edges}
            for port in node.get_input_ports():
                if port.name not in covered_in:
                    self._exposed_inputs.append({"node_id": nid, "port_name": port.name})

            out_edges = self.get_edges_out(nid)
            covered_out = {e.source_port for e in out_edges}
            for port in node.get_output_ports():
                if port.name not in covered_out:
                    self._exposed_outputs.append({"node_id": nid, "port_name": port.name})
        self._execution_version += 1

    # --- state dict (preparation for Phase 5) ----------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict, deduplicated by block_id for shared blocks."""
        result: Dict[str, Any] = {}
        seen_block_ids: Dict[str, str] = {}
        aliases: Dict[str, str] = {}
        for nid in sorted(self._nodes.keys()):
            node = self._nodes[nid]
            if not hasattr(node, "state_dict"):
                continue
            sd = node.state_dict()
            if not sd:
                continue
            bid = getattr(node, "block_id", nid)
            if bid in seen_block_ids:
                aliases[nid] = seen_block_ids[bid]
            else:
                seen_block_ids[bid] = nid
                result[nid] = sd
        if aliases:
            result["_aliases"] = aliases
        return result

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        aliases: Dict[str, str] = state.get("_aliases", {})
        expanded = {k: v for k, v in state.items() if k != "_aliases"}
        for nid, alias_target in aliases.items():
            if alias_target in expanded:
                expanded[nid] = expanded[alias_target]

        if strict:
            extra = set(expanded.keys()) - set(self._nodes.keys())
            if extra:
                raise KeyError(f"state_dict has keys not in graph: {extra}")
        for nid, node in self._nodes.items():
            if nid in expanded and hasattr(node, "load_state_dict"):
                node.load_state_dict(expanded[nid], strict=strict)

    # --- device / trainable ----------------------------------------------

    def to(self, device: Any) -> "Hypergraph":
        for node in self._nodes.values():
            if hasattr(node, "to") and callable(node.to):
                node.to(device)
        return self

    def set_trainable(self, node_id: str, trainable: bool) -> None:
        if node_id not in self._nodes:
            raise ValueError(f"Node '{node_id}' not in graph")
        self._node_trainable[node_id] = trainable

    def trainable_parameters(self) -> Iterator[Any]:
        """Yield trainable parameters, deduplicated by block_id."""
        seen_block_ids: Set[str] = set()
        for nid, node in self._nodes.items():
            if not self._node_trainable.get(nid, True):
                continue
            block_id = getattr(node, "block_id", nid)
            if block_id in seen_block_ids:
                continue
            seen_block_ids.add(block_id)
            if hasattr(node, "trainable_parameters") and callable(node.trainable_parameters):
                yield from node.trainable_parameters()

    # --- serialization (Phase 5) delegates to hypergraph.serialization ----

    def save(
        self,
        directory: "str | Path",
        *,
        config_filename: str = "config.json",
        checkpoint_filename: str = "checkpoint.pkl",
    ) -> Path:
        from yggdrasill.hypergraph.serialization import save_hypergraph
        return save_hypergraph(
            self, directory,
            config_filename=config_filename,
            checkpoint_filename=checkpoint_filename,
        )

    def save_config(
        self,
        directory: "str | Path",
        *,
        filename: str = "config.json",
    ) -> Path:
        from yggdrasill.hypergraph.serialization import save_config as _save_cfg
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        cfg = self.to_config()
        cfg["schema_version"] = "1.0"
        _save_cfg(cfg, d / filename)
        return d

    def save_checkpoint(
        self,
        directory: "str | Path",
        *,
        filename: str = "checkpoint.pkl",
    ) -> Path:
        from yggdrasill.hypergraph.serialization import save_checkpoint as _save_ckpt
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        from yggdrasill.hypergraph.serialization import _deduplicate_state
        full_state = self.state_dict()
        deduped = _deduplicate_state(self, full_state)
        _save_ckpt(deduped, d / filename)
        return d

    @classmethod
    def load(
        cls,
        directory: "str | Path",
        *,
        registry: Optional[Any] = None,
        validate: bool = False,
        config_filename: str = "config.json",
        checkpoint_filename: str = "checkpoint.pkl",
        load_checkpoint: bool = True,
    ) -> "Hypergraph":
        from yggdrasill.hypergraph.serialization import load_hypergraph
        return load_hypergraph(
            directory, registry=registry, validate=validate,
            config_filename=config_filename,
            checkpoint_filename=checkpoint_filename,
            load_checkpoint_flag=load_checkpoint,
        )

    @classmethod
    def load_config(
        cls,
        directory: "str | Path",
        *,
        registry: Optional[Any] = None,
        validate: bool = False,
        config_filename: str = "config.json",
    ) -> "Hypergraph":
        from yggdrasill.hypergraph.serialization import load_hypergraph
        return load_hypergraph(
            directory, registry=registry, validate=validate,
            config_filename=config_filename,
            load_checkpoint_flag=False,
        )

    def load_from_checkpoint(
        self,
        directory: "str | Path",
        *,
        checkpoint_filename: str = "checkpoint.pkl",
    ) -> None:
        """Load weights into an already-built hypergraph."""
        from yggdrasill.hypergraph.serialization import (
            _expand_deduped_state, _read_checkpoint,
        )
        p = Path(directory) / checkpoint_filename
        state = _read_checkpoint(p)
        expanded = _expand_deduped_state(self, state)
        self.load_state_dict(expanded, strict=False)

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _spec_key(entry: Dict[str, Any]) -> str:
        if "name" in entry and entry["name"] is not None:
            return entry["name"]
        nid = entry.get("node_id") or entry.get("graph_id", "")
        return f"{nid}:{entry['port_name']}"
