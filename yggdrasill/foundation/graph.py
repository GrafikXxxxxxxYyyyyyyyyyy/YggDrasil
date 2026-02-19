"""
Graph: container of nodes and edges; structure + serialization.

Canon: WorldGenerator_2.0/TODO_01 §4, TODO_03 (contract: exposed inputs/outputs).
- Nodes (node_id -> Node), edges (source_node, source_port, target_node, target_port)
- Exposed inputs/outputs for executor and pipeline (get_input_spec, get_output_spec)
- Serialize structure (config) and weights (checkpoint); load_from_checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Union

from yggdrasill.foundation.block import AbstractBaseBlock
from yggdrasill.foundation.node import Node
from yggdrasill.foundation.port import Port
from yggdrasill.foundation.registry import BlockRegistry

# Schema version for config roundtrip and future migrations (TODO_03 §5.1)
GRAPH_CONFIG_SCHEMA_VERSION = "1.0"


@dataclass
class ValidationResult:
    """Result of graph validation: errors (blocking) and warnings (optional). TODO_03 §3.3."""

    errors: List[str]
    warnings: List[str]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


@dataclass(frozen=True)
class Edge:
    """Single edge: (source_node, source_port) -> (target_node, target_port)."""

    source_node: str
    source_port: str
    target_node: str
    target_port: str

    def __post_init__(self) -> None:
        for name in ("source_node", "source_port", "target_node", "target_port"):
            v = getattr(self, name)
            if not v or (isinstance(v, str) and not v.strip()):
                raise ValueError(f"{name} must be non-empty")


def _default_node_id(block: AbstractBaseBlock, nodes: Dict[str, Any]) -> str:
    """Уникальный node_id из block_type и числа узлов (TODO_02 B.2.1)."""
    base = (block.block_type or "block").replace("/", "_").replace(" ", "_")
    n = len(nodes)
    candidate = f"{base}_{n}"
    while candidate in nodes:
        n += 1
        candidate = f"{base}_{n}"
    return candidate


def _load_pretrained_into_block(block: AbstractBaseBlock, pretrained: Union[str, Dict[str, Any]]) -> None:
    """Загрузить веса в блок: путь к JSON-чекпоинту или state_dict."""
    if isinstance(pretrained, str):
        import json
        with open(pretrained, "r", encoding="utf-8") as f:
            pretrained = json.load(f)
    block.load_state_dict(pretrained, strict=False)


def _ensure_auto_connect(graph: "Graph", node_id: str, block: AbstractBaseBlock) -> None:
    """Включить автосвязывание по ролям (при необходимости подтянуть task_nodes) и применить к новому узлу."""
    if not getattr(graph, "auto_connect_fn", None):
        try:
            from yggdrasill.task_nodes import use_task_node_auto_connect
            use_task_node_auto_connect(graph)
        except ImportError:
            pass
    if getattr(graph, "auto_connect_fn", None):
        graph.auto_connect_fn(graph, node_id, block)


def _exposed_spec_entry(node_id: str, port_name: str, name: Optional[str] = None) -> Dict[str, Any]:
    """One entry for exposed_inputs or exposed_outputs in config/spec."""
    out: Dict[str, Any] = {"node_id": node_id, "port_name": port_name}
    if name is not None:
        out["name"] = name
    return out


class Graph:
    """
    Graph = nodes (node_id -> Node) + edges + exposed inputs/outputs.
    Supports add_node, add_edge, expose_input/expose_output (contract for executor/pipeline),
    config serialization with schema_version, checkpoint save/load.
    """

    _template_builders: Dict[str, Callable[..., Dict[str, Any]]] = {}

    def __init__(self, graph_id: Optional[str] = None) -> None:
        self._graph_id = graph_id or "graph"
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Edge] = []
        self._in_edges_by_node: Dict[str, List[Edge]] = {}
        self._out_edges_by_node: Dict[str, List[Edge]] = {}
        self._exposed_inputs: List[Dict[str, Any]] = []   # {"node_id", "port_name", "name"?}
        self._exposed_outputs: List[Dict[str, Any]] = []
        # Обучаемость: по умолчанию все узлы trainable (TRAINABILITY_AT_ALL_LEVELS, TODO_02 B.5)
        self._node_trainable: Dict[str, bool] = {}
        # Метаданные для executor/пайплайна: тип задачи (diffusion, llm, vlm, generic), произвольные поля
        self._graph_kind: Optional[str] = None   # "diffusion" | "llm" | "vlm" | "data" | "generic"
        self._metadata: Dict[str, Any] = {}
        # Bump on structural change so executor can invalidate execution plan cache (TODO_03)
        self._execution_version: int = 0

    @property
    def graph_id(self) -> str:
        return self._graph_id

    @property
    def node_ids(self) -> Set[str]:
        return set(self._nodes)

    @property
    def graph_kind(self) -> Optional[str]:
        """Тип задачи графа для executor/пайплайна: diffusion, llm, vlm, data, generic."""
        return self._graph_kind

    @graph_kind.setter
    def graph_kind(self, value: Optional[str]) -> None:
        self._graph_kind = value

    @property
    def metadata(self) -> Dict[str, Any]:
        """Произвольные метаданные (версия, описание, параметры run и т.д.)."""
        return self._metadata

    def add_node(
        self,
        node_id_or_block_or_type: Union[str, Node, AbstractBaseBlock],
        block_type: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        pretrained: Optional[Union[str, Dict[str, Any]]] = None,
        registry: Optional[BlockRegistry] = None,
        auto_connect: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Добавить узел в граф. Возвращает node_id.

        Простой вариант (в две строчки):
            g.add_node("MyBackbone", "backbone/identity")
            g.add_node("MySolver", "solver/identity", pretrained="path/to.ckpt")

        - Два аргумента (node_id, block_type): собрать блок из реестра, добавить с заданным именем.
          auto_connect=True по умолчанию (автосвязывание по ролям). pretrained — путь к чекпоинту или state_dict.
        - Один аргумент Node: добавить готовый узел (для from_config и тестов).
        - Один аргумент блок или block_type: добавить с сгенерированным node_id (обратная совместимость).
        """
        # Один аргумент — Node, блок или block_type
        if block_type is None:
            if isinstance(node_id_or_block_or_type, Node) and (config or pretrained or registry is not None or kwargs):
                raise ValueError(
                    "При передаче Node не передавайте config, pretrained, registry или другие аргументы"
                )
            return self._add_node_legacy(
                node_id_or_block_or_type,
                config=config,
                registry=registry,
                auto_connect=auto_connect,
            )

        # Два аргумента: node_id, block_type — основной сценарий
        nid = node_id_or_block_or_type
        if not isinstance(nid, str) or not nid.strip():
            raise ValueError("node_id должен быть непустой строкой")
        if not isinstance(block_type, str) or not block_type.strip():
            raise ValueError("block_type должен быть непустой строкой")

        reg = registry or BlockRegistry.global_registry()
        build_config = {"block_type": block_type, **(config or {})}
        block = reg.build(build_config)

        if nid in self._nodes:
            raise ValueError(f"Узел уже существует: {nid}")
        self.add_node(Node(nid, block))
        self._node_trainable[nid] = kwargs.get("trainable", True)

        if pretrained is not None:
            _load_pretrained_into_block(block, pretrained)

        if auto_connect:
            _ensure_auto_connect(self, nid, block)
        return nid

    def _add_node_legacy(
        self,
        block_or_type: Union[Node, AbstractBaseBlock, str],
        *,
        config: Optional[Dict[str, Any]] = None,
        registry: Optional[BlockRegistry] = None,
        auto_connect: bool = True,
    ) -> str:
        """Обратная совместимость: один аргумент — Node, блок или block_type."""
        if isinstance(block_or_type, Node):
            if config is not None or registry is not None:
                raise ValueError("При передаче Node не передавайте config или registry")
            if block_or_type.node_id in self._nodes:
                raise ValueError(f"Узел уже существует: {block_or_type.node_id}")
            self._nodes[block_or_type.node_id] = block_or_type
            self._in_edges_by_node.setdefault(block_or_type.node_id, [])
            self._out_edges_by_node.setdefault(block_or_type.node_id, [])
            self._execution_version += 1
            return block_or_type.node_id

        if isinstance(block_or_type, str):
            reg = registry or BlockRegistry.global_registry()
            build_config = {"block_type": block_or_type, **(config or {})}
            block = reg.build(build_config)
        else:
            block = block_or_type

        nid = _default_node_id(block, self._nodes)
        if nid in self._nodes:
            raise ValueError(f"Узел уже существует: {nid}")
        self.add_node(Node(nid, block))
        self._node_trainable[nid] = True
        if auto_connect:
            _ensure_auto_connect(self, nid, block)
        return nid

    def add_edge(self, edge: Edge) -> None:
        self._validate_edge(edge)
        if edge in self._edges:
            return  # idempotent
        self._edges.append(edge)
        self._in_edges_by_node.setdefault(edge.target_node, []).append(edge)
        self._out_edges_by_node.setdefault(edge.source_node, []).append(edge)
        self._execution_version += 1

    def expose_input(self, node_id: str, port_name: str, name: Optional[str] = None) -> None:
        """Mark an input port as graph-level input (for executor / pipeline contract)."""
        if node_id not in self._nodes:
            raise ValueError(f"Node not found: {node_id}")
        port = self._nodes[node_id].block.get_port(port_name)
        if port is None or not port.is_input:
            raise ValueError(f"Input port {port_name!r} not found on node {node_id}")
        self._exposed_inputs.append(_exposed_spec_entry(node_id, port_name, name))

    def expose_output(self, node_id: str, port_name: str, name: Optional[str] = None) -> None:
        """Mark an output port as graph-level output (for executor / pipeline contract)."""
        if node_id not in self._nodes:
            raise ValueError(f"Node not found: {node_id}")
        port = self._nodes[node_id].block.get_port(port_name)
        if port is None or not port.is_output:
            raise ValueError(f"Output port {port_name!r} not found on node {node_id}")
        self._exposed_outputs.append(_exposed_spec_entry(node_id, port_name, name))

    def infer_exposed_ports(self) -> None:
        """
        Set exposed inputs/outputs by inference: input = port with no incoming edge from another node;
        output = port with no outgoing edge to another node (canon §2.3 optional).
        Use when no explicit expose_input/expose_output was called; overwrites current _exposed_inputs/_exposed_outputs.
        """
        self._exposed_inputs = []
        self._exposed_outputs = []
        for nid, node in self._nodes.items():
            in_ports_covered = {e.target_port for e in self.get_edges_in(nid)}
            for port in node.block.get_input_ports():
                if port.name not in in_ports_covered:
                    self._exposed_inputs.append(_exposed_spec_entry(nid, port.name, None))
            out_ports_covered = {e.source_port for e in self.get_edges_out(nid)}
            for port in node.block.get_output_ports():
                if port.name not in out_ports_covered:
                    self._exposed_outputs.append(_exposed_spec_entry(nid, port.name, None))

    def get_input_spec(self, include_dtype: bool = False) -> List[Dict[str, Any]]:
        """
        Contract: list of {node_id, port_name, name?, dtype?} for run(inputs) keys.
        If include_dtype=True, add dtype from port (for multi-endpoint and pipeline type checks).
        """
        if not include_dtype:
            return list(self._exposed_inputs)
        out = []
        for e in self._exposed_inputs:
            entry = dict(e)
            node = self.get_node(entry["node_id"])
            if node is not None:
                port = node.block.get_port(entry["port_name"])
                if port is not None:
                    entry["dtype"] = getattr(port.dtype, "name", str(port.dtype))
            out.append(entry)
        return out

    def get_output_spec(self, include_dtype: bool = False) -> List[Dict[str, Any]]:
        """
        Contract: list of {node_id, port_name, name?, dtype?} for run() return keys.
        If include_dtype=True, add dtype from port (for multi-endpoint and pipeline type checks).
        """
        if not include_dtype:
            return list(self._exposed_outputs)
        out = []
        for e in self._exposed_outputs:
            entry = dict(e)
            node = self.get_node(entry["node_id"])
            if node is not None:
                port = node.block.get_port(entry["port_name"])
                if port is not None:
                    entry["dtype"] = getattr(port.dtype, "name", str(port.dtype))
            out.append(entry)
        return out

    def _validate_edge(self, edge: Edge) -> None:
        if edge.source_node not in self._nodes:
            raise ValueError(f"Source node not found: {edge.source_node}")
        if edge.target_node not in self._nodes:
            raise ValueError(f"Target node not found: {edge.target_node}")
        sn = self._nodes[edge.source_node].block
        tn = self._nodes[edge.target_node].block
        sp = sn.get_port(edge.source_port)
        tp = tn.get_port(edge.target_port)
        if sp is None:
            raise ValueError(f"Source port {edge.source_port!r} not found on node {edge.source_node}")
        if tp is None:
            raise ValueError(f"Target port {edge.target_port!r} not found on node {edge.target_node}")
        if not sp.compatible_with(tp):
            raise ValueError(
                f"Ports incompatible: ({edge.source_node}.{edge.source_port}) -> "
                f"({edge.target_node}.{edge.target_port})"
            )

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def get_edges_in(self, node_id: str) -> List[Edge]:
        return list(self._in_edges_by_node.get(node_id, []))

    def get_edges_out(self, node_id: str) -> List[Edge]:
        return list(self._out_edges_by_node.get(node_id, []))

    def get_edges(self) -> List[Edge]:
        return list(self._edges)

    def trainable_parameters(self) -> Iterator[Any]:
        """
        Параметры для оптимизатора: только узлы с trainable=True (TRAINABILITY_AT_ALL_LEVELS, TODO_02 B.5).
        """
        for nid, node in self._nodes.items():
            if self._node_trainable.get(nid, True):
                yield from node.block.trainable_parameters()

    def set_trainable(self, node_id: str, trainable: bool) -> None:
        """Пометить узел как обучаемый или замороженный."""
        if node_id not in self._nodes:
            raise ValueError(f"Узел не найден: {node_id}")
        self._node_trainable[node_id] = trainable

    def validate(self, strict: bool = True) -> ValidationResult:
        """
        Validate graph: required ports, optionally reachability (warnings).
        If strict=True and there are errors, raise ValueError; otherwise return ValidationResult.
        """
        errors: List[str] = []
        warnings: List[str] = []
        exposed_set: Set[tuple[str, str]] = {
            (e["node_id"], e["port_name"]) for e in self._exposed_inputs
        }
        for nid, node in self._nodes.items():
            for port in node.block.get_input_ports():
                if port.optional:
                    continue
                if (nid, port.name) in exposed_set:
                    continue
                incoming = [e for e in self.get_edges_in(nid) if e.target_port == port.name]
                if not incoming:
                    msg = f"Required input port {nid}.{port.name} has no incoming edge and is not exposed"
                    errors.append(msg)
        # Optional: reachability — nodes reachable from exposed inputs and leading to exposed outputs
        if self._nodes and (self._exposed_inputs or self._exposed_outputs):
            reachable_from_inputs = self._reachable_from(exposed_inputs=True)
            reaches_outputs = self._reaches_exposed_outputs()
            for nid in self.node_ids:
                if nid not in reachable_from_inputs and self._exposed_inputs:
                    warnings.append(f"Node {nid} is not reachable from any exposed input")
                if nid not in reaches_outputs and self._exposed_outputs:
                    warnings.append(f"Node {nid} does not lead to any exposed output")
        result = ValidationResult(errors=errors, warnings=warnings)
        if strict and result.errors:
            raise ValueError("; ".join(result.errors))
        return result

    def _reachable_from(self, *, exposed_inputs: bool) -> Set[str]:
        """Nodes reachable from exposed input nodes (BFS following out-edges)."""
        if not exposed_inputs:
            return set()
        start = {e["node_id"] for e in self._exposed_inputs}
        reachable: Set[str] = set(start)
        queue = list(start)
        while queue:
            nid = queue.pop(0)
            for e in self.get_edges_out(nid):
                if e.target_node not in reachable:
                    reachable.add(e.target_node)
                    queue.append(e.target_node)
        return reachable

    def _reaches_exposed_outputs(self) -> Set[str]:
        """Nodes from which an exposed output is reachable (BFS backward on in-edges)."""
        if not self._exposed_outputs:
            return set()
        start = {e["node_id"] for e in self._exposed_outputs}
        reaches: Set[str] = set(start)
        queue = list(start)
        while queue:
            nid = queue.pop(0)
            for e in self.get_edges_in(nid):
                if e.source_node not in reaches:
                    reaches.add(e.source_node)
                    queue.append(e.source_node)
        return reaches

    # --- Execution (TODO_03) ---

    def run(
        self,
        inputs: Dict[str, Any],
        *,
        training: bool = False,
        num_loop_steps: Optional[int] = None,
        device: Any = None,
        callbacks: Optional[List[Callable[..., None]]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Execute graph; delegates to executor.run(self, inputs, ...). Canon: TODO_03 §4.2, §9.2."""
        from yggdrasill.executor import run as _run
        return _run(
            self, inputs,
            training=training, num_loop_steps=num_loop_steps, device=device,
            callbacks=callbacks, dry_run=dry_run,
        )

    def to(self, device: Any) -> "Graph":
        """Move all blocks that support .to(device) to the given device. Returns self for chaining."""
        for nid in self.node_ids:
            node = self.get_node(nid)
            if node is not None and hasattr(node.block, "to") and callable(getattr(node.block, "to")):
                node.block.to(device)
        return self

    # --- Serialization: structure (config) ---

    def to_config(self) -> Dict[str, Any]:
        """
        Сериализация структуры графа: schema_version, nodes (с trainable), edges, exposed_*, graph_kind, metadata.
        Без весов; по конфигу можно собрать граф через реестр (TODO_03 §5.1).
        """
        nodes_cfg = []
        for nid, node in self._nodes.items():
            block = node.block
            nodes_cfg.append({
                "node_id": nid,
                "block_type": block.block_type,
                "block_id": block.block_id,
                "config": block.config,
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
            "schema_version": GRAPH_CONFIG_SCHEMA_VERSION,
            "graph_id": self._graph_id,
            "nodes": nodes_cfg,
            "edges": edges_cfg,
            "exposed_inputs": list(self._exposed_inputs),
            "exposed_outputs": list(self._exposed_outputs),
        }
        if self._graph_kind is not None:
            out["graph_kind"] = self._graph_kind
        if self._metadata:
            out["metadata"] = dict(self._metadata)
        return out

    @classmethod
    def from_template(
        cls,
        template_name: str,
        *,
        registry: Optional[BlockRegistry] = None,
        validate: bool = False,
        **kwargs: Any,
    ) -> "Graph":
        """
        Build graph from a named template (e.g. "text_to_image").
        Templates are registered by extensions (e.g. task_nodes); builder returns config dict.
        If validate=True, call graph.validate(strict=True) after build.
        """
        builders = getattr(cls, "_template_builders", None) or {}
        if template_name not in builders:
            raise KeyError(
                f"Unknown template: {template_name!r}. Known: {list(builders)}"
            )
        config = builders[template_name](**kwargs)
        return cls.from_config(config, registry=registry, validate=validate)

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        registry: Optional[BlockRegistry] = None,
        validate: bool = False,
    ) -> Graph:
        """
        Build graph from config: create blocks via registry, edges, then exposed_inputs/outputs.
        If validate=True, call g.validate(strict=True) after build (canon §7.3).
        Node config may use {"ref": "path/to/file.yaml"} to load config from file (canon §7.1).
        """
        reg = registry or BlockRegistry.global_registry()
        g = cls(graph_id=config.get("graph_id", "graph"))
        g._graph_kind = config.get("graph_kind")
        g._metadata = dict(config.get("metadata", {}))
        # Create nodes
        for nc in config.get("nodes", []):
            raw_cfg = nc.get("config", {})
            node_cfg = cls._resolve_config_ref(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
            if not isinstance(node_cfg, dict):
                node_cfg = {}
            block_cfg = {
                "block_type": nc["block_type"],
                "block_id": nc.get("block_id"),
                **node_cfg,
            }
            block = reg.build(block_cfg)
            nid = nc["node_id"]
            g.add_node(Node(node_id=nid, block=block))
            g._node_trainable[nid] = nc.get("trainable", True)
        # Add edges
        for ec in config.get("edges", []):
            g.add_edge(Edge(
                source_node=ec["source_node"],
                source_port=ec["source_port"],
                target_node=ec["target_node"],
                target_port=ec["target_port"],
            ))
        # Restore exposed inputs/outputs
        for entry in config.get("exposed_inputs", []):
            g._exposed_inputs.append(_exposed_spec_entry(
                entry["node_id"], entry["port_name"], entry.get("name")
            ))
        for entry in config.get("exposed_outputs", []):
            g._exposed_outputs.append(_exposed_spec_entry(
                entry["node_id"], entry["port_name"], entry.get("name")
            ))
        if validate:
            g.validate(strict=True)
        return g

    @staticmethod
    def _resolve_config_ref(config: Dict[str, Any]) -> Dict[str, Any]:
        """If config is {\"ref\": \"path/to/file\"}, load and return file contents; else return config."""
        if not config or set(config) != {"ref"}:
            return config
        path = config["ref"]
        try:
            from omegaconf import OmegaConf
            loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        except Exception:
            import json
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        return loaded if isinstance(loaded, dict) else {}

    # --- Serialization: weights (checkpoint) ---

    def state_dict(self) -> Dict[str, Any]:
        """Aggregated state_dict of all blocks, keyed by node_id (or block_id)."""
        out: Dict[str, Any] = {}
        for nid, node in self._nodes.items():
            sd = node.block.state_dict()
            if sd:
                out[nid] = sd
        return out

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        """Load state into blocks by node_id."""
        for nid, node in self._nodes.items():
            if nid in state:
                node.block.load_state_dict(state[nid], strict=strict)
        if strict:
            for key in state:
                if key not in self._nodes:
                    raise KeyError(f"Unknown node_id in checkpoint: {key}")

    def load_from_checkpoint(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint: Optional[Dict[str, Any]] = None,
        registry: Optional[BlockRegistry] = None,
        backend: str = "json",
        **kwargs: Any,
    ) -> None:
        """
        Load graph: optionally rebuild from config, then load weights.
        Either (config_path or config) and optionally (checkpoint_path or checkpoint_dir or checkpoint).
        checkpoint_dir: directory with node_id.json per node (canon §6.1 variant B).
        backend: "json" (default; only supported for now). Future: "torch", "safetensors".
        """
        import json
        import os
        if backend != "json":
            raise NotImplementedError(
                f"Checkpoint backend {backend!r} not implemented; only 'json' is supported. "
                "Future backends: 'torch', 'safetensors'."
            )
        if config_path is not None:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        if config is not None:
            new_g = Graph.from_config(config, registry=registry)
            self._graph_id = new_g._graph_id
            self._nodes = new_g._nodes
            self._edges = new_g._edges
            self._in_edges_by_node = new_g._in_edges_by_node
            self._out_edges_by_node = new_g._out_edges_by_node
            self._exposed_inputs = new_g._exposed_inputs
            self._exposed_outputs = new_g._exposed_outputs
        if checkpoint_dir is not None:
            state = {}
            for fname in os.listdir(checkpoint_dir):
                if fname.endswith(".json"):
                    nid = fname[:-5]
                    with open(os.path.join(checkpoint_dir, fname), "r", encoding="utf-8") as f:
                        state[nid] = json.load(f)
            if state:
                self.load_state_dict(state, strict=False)
        elif checkpoint_path is not None:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                self.load_state_dict(json.load(f), strict=False)
        elif checkpoint is not None:
            self.load_state_dict(checkpoint, strict=False)

    def save_config(self, path: str) -> None:
        """
        Write graph structure (to_config) to a JSON file.
        Canon: TODO_02 B.4, SERIALIZATION_AT_ALL_LEVELS. Use with save_checkpoint for full save.
        """
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_config(), f, indent=2)

    def save_checkpoint(
        self,
        path: str,
        format: str = "single",
        backend: str = "json",
    ) -> None:
        """
        Write graph weights (state_dict) to file(s).
        format: "single" (default) — one file at path; "dir" — path is a directory, one file per node (canon §6.1–6.2).
        backend: "json" (default; only supported for now). Future: "torch", "safetensors" for tensor state.
        """
        if backend != "json":
            raise NotImplementedError(
                f"Checkpoint backend {backend!r} not implemented; only 'json' is supported. "
                "Future backends: 'torch', 'safetensors'."
            )
        import json
        import os
        state = self.state_dict()
        if format == "dir":
            os.makedirs(path, exist_ok=True)
            for nid, sd in state.items():
                if os.path.sep in nid or (os.path.altsep and os.path.altsep in nid):
                    raise ValueError(f"Node_id {nid!r} cannot contain path separators when using format='dir'")
                with open(os.path.join(path, f"{nid}.json"), "w", encoding="utf-8") as f:
                    json.dump(sd, f, indent=2)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

    def save(self, save_dir: str) -> None:
        """
        Save graph to a directory: config and checkpoint (TODO_03).
        Writes save_dir/config.json and save_dir/checkpoint.json (or .yaml for config if preferred).
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        self.save_config(os.path.join(save_dir, "config.json"))
        self.save_checkpoint(os.path.join(save_dir, "checkpoint.json"))

    @classmethod
    def load(
        cls,
        save_dir: str,
        *,
        registry: Optional[BlockRegistry] = None,
    ) -> "Graph":
        """
        Load graph from a directory (config + checkpoint). Returns new Graph instance.
        Expects save_dir/config.json (or config.yaml) and save_dir/checkpoint.json.
        """
        import json
        import os
        yaml_path = os.path.join(save_dir, "config.yaml")
        json_path = os.path.join(save_dir, "config.json")
        checkpoint_path = os.path.join(save_dir, "checkpoint.json")
        if os.path.isfile(yaml_path):
            g = cls.from_yaml(yaml_path, registry=registry)
        elif os.path.isfile(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                g = cls.from_config(json.load(f), registry=registry)
        else:
            raise FileNotFoundError(f"No config found in {save_dir} (config.json or config.yaml)")
        if os.path.isfile(checkpoint_path):
            g.load_from_checkpoint(checkpoint_path=checkpoint_path)
        return g

    @classmethod
    def from_yaml(cls, path: str, registry: Optional[BlockRegistry] = None) -> "Graph":
        """
        Load graph config from a YAML or JSON file and build via from_config.
        Canon: TODO_02 B.6, TODO_03. Uses OmegaConf if available for YAML; falls back to JSON.
        """
        try:
            from omegaconf import OmegaConf
            config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        except Exception:
            import json
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
        return cls.from_config(config, registry=registry)
