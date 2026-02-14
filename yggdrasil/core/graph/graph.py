# yggdrasil/core/graph/graph.py
"""ComputeGraph — направленный ациклический граф из блоков.

Это главная структура данных YggDrasil v2.
Pipeline = граф из блоков, соединённых через порты.
"""
from __future__ import annotations

import copy
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Set, Tuple

from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Edge — ребро графа
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Edge:
    """Соединение между выходным портом одного узла и входным портом другого.
    
    Attributes:
        src_node: Имя узла-источника.
        src_port: Имя выходного порта источника.
        dst_node: Имя узла-приёмника.
        dst_port: Имя входного порта приёмника.
    """
    src_node: str
    src_port: str
    dst_node: str
    dst_port: str

    def __repr__(self) -> str:
        return f"{self.src_node}.{self.src_port} -> {self.dst_node}.{self.dst_port}"


# ---------------------------------------------------------------------------
# ComputeGraph — DAG
# ---------------------------------------------------------------------------

class ComputeGraph:
    """Направленный ациклический граф блоков — настоящий Lego-конструктор.
    
    Позволяет:
    - Добавлять/удалять/заменять узлы
    - Соединять порты блоков рёбрами
    - Объявлять входы/выходы графа
    - Валидировать корректность
    - Выполнять граф (через GraphExecutor)
    - Сериализовать в YAML и загружать обратно
    - Визуализировать как Mermaid-диаграмму
    
    Пример::
    
        graph = ComputeGraph("sd15_txt2img")
        graph.add_node("clip", clip_block)
        graph.add_node("unet", unet_block)
        graph.connect("clip", "embedding", "unet", "condition")
        graph.expose_input("prompt", "clip", "text")
        graph.expose_output("noise_pred", "unet", "output")
    """
    
    def __init__(self, name: str = "unnamed"):
        self.name: str = name
        self.nodes: OrderedDict[str, Any] = OrderedDict()  # name -> AbstractBlock
        self.edges: List[Edge] = []
        # Fan-out: one graph input can feed multiple (node, port) targets
        self.graph_inputs: Dict[str, List[Tuple[str, str]]] = {}   # input_name -> [(node, port), ...]
        self.graph_outputs: Dict[str, Tuple[str, str]] = {}        # output_name -> (node, port)
        self.metadata: Dict[str, Any] = {}
        # Device tracking
        self._device: Any = None
        self._dtype: Any = None
    
    # ==================== DEVICE MANAGEMENT ====================
    
    def to(self, device=None, dtype=None) -> ComputeGraph:
        """Перенести весь граф на устройство.
        
        Рекурсивно переносит все узлы, включая вложенные SubGraph.
        
        Когда dtype=None (по умолчанию), каждый блок сохраняет свой
        оригинальный dtype. Это позволяет UNet оставаться в float16
        на MPS (быстрее и экономичнее), а CLIP — в float32
        (нужен для LayerNorm).
        
        Args:
            device: Устройство ("cuda", "mps", "cpu", torch.device).
            dtype: Тип данных. Если None — блоки сохраняют свой dtype.
                   Если указан — все блоки конвертируются.
        
        Returns:
            self (для chaining).
        """
        import torch
        
        if isinstance(device, str):
            device = torch.device(device)
        
        self._device = device
        self._dtype = dtype  # None means "keep original per-block dtype"
        
        # Move all nodes recursively
        for name, block in self.nodes.items():
            self._move_block(block, device, dtype)
            # Recurse into SubGraphs (LoopSubGraph, SubGraph)
            if hasattr(block, 'graph') and block.graph is not None:
                block.graph.to(device, dtype)
        
        return self
    
    @staticmethod
    def _move_block(block, device, dtype):
        """Move a single block to device/dtype."""
        import logging
        _logger = logging.getLogger(__name__)
        
        if not hasattr(block, 'to'):
            return
        try:
            if dtype is not None:
                block.to(device=device, dtype=dtype)
            else:
                block.to(device)
        except TypeError:
            # Some blocks don't accept dtype kwarg — retry with device only
            try:
                block.to(device)
            except Exception as e:
                _logger.warning(
                    f"Failed to move block {getattr(block, 'block_type', type(block).__name__)} "
                    f"to {device}: {e}"
                )
        except Exception as e:
            _logger.warning(
                f"Failed to move block {getattr(block, 'block_type', type(block).__name__)} "
                f"to {device}/{dtype}: {e}"
            )
    
    @property
    def device(self):
        """Текущее устройство графа."""
        return self._device
    
    # ==================== СБОРКА ГРАФА ====================
    
    def add_node(self, name: str, block: Any) -> ComputeGraph:
        """Добавить узел (блок) в граф.
        
        Args:
            name: Уникальное имя узла в графе.
            block: Экземпляр AbstractBlock.
        
        Returns:
            self (для chaining).
        """
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists in graph '{self.name}'")
        self.nodes[name] = block
        return self
    
    def connect(
        self,
        src: str,
        src_port: str,
        dst: str,
        dst_port: str,
        *,
        validate: bool = True,
    ) -> ComputeGraph:
        """Соединить выходной порт одного узла с входным портом другого.
        
        Args:
            src: Имя узла-источника.
            src_port: Имя выходного порта.
            dst: Имя узла-приёмника.
            dst_port: Имя входного порта.
            validate: Проверять ли совместимость портов (default True).
        
        Returns:
            self (для chaining).
        
        Raises:
            ValueError: Если узлы не найдены или порты несовместимы.
        """
        if src not in self.nodes:
            raise ValueError(f"Source node '{src}' not found in graph '{self.name}'")
        if dst not in self.nodes:
            raise ValueError(f"Destination node '{dst}' not found in graph '{self.name}'")
        
        # Port validation (soft — warns if ports declared but incompatible)
        if validate:
            self._validate_edge(src, src_port, dst, dst_port)
        
        edge = Edge(src_node=src, src_port=src_port, dst_node=dst, dst_port=dst_port)
        self.edges.append(edge)
        return self
    
    def _validate_edge(self, src: str, src_port: str, dst: str, dst_port: str):
        """Validate port compatibility for an edge.
        
        Raises ValueError if validation_mode='strict'.
        Otherwise logs a warning.
        """
        import logging
        _logger = logging.getLogger(__name__)
        
        src_block = self.nodes[src]
        dst_block = self.nodes[dst]
        
        src_ports = getattr(src_block, 'declare_io', lambda: {})()
        dst_ports = getattr(dst_block, 'declare_io', lambda: {})()
        
        if not src_ports or not dst_ports:
            return
        
        sp = src_ports.get(src_port)
        dp = dst_ports.get(dst_port)
        
        if sp is not None and dp is not None:
            from yggdrasil.core.block.port import PortValidator
            valid, msg = PortValidator.validate_connection(
                getattr(src_block, 'block_type', 'unknown'), sp,
                getattr(dst_block, 'block_type', 'unknown'), dp,
            )
            if not valid:
                if self.metadata.get("strict_validation", False):
                    raise ValueError(f"Port incompatible: {msg}")
                _logger.warning(f"Port compatibility warning: {msg}")
        elif sp is None and src_ports:
            msg = f"Output port '{src_port}' not declared on {src} ({getattr(src_block, 'block_type', '?')})"
            if self.metadata.get("strict_validation", False):
                raise ValueError(msg)
            _logger.debug(msg)
        elif dp is None and dst_ports:
            msg = f"Input port '{dst_port}' not declared on {dst} ({getattr(dst_block, 'block_type', '?')})"
            if self.metadata.get("strict_validation", False):
                raise ValueError(msg)
            _logger.debug(msg)
    
    def expose_input(
        self,
        graph_input_name: str,
        node: str,
        port: str,
    ) -> ComputeGraph:
        """Объявить входной порт графа (маппинг на входной порт узла).
        
        При execute(**inputs) значение inputs[graph_input_name]
        будет передано в node.port.
        
        Поддерживает fan-out: один и тот же graph_input_name может быть
        привязан к нескольким (node, port) парам. Повторный вызов с тем же
        именем добавляет новый target, а не перезаписывает старый.
        """
        if node not in self.nodes:
            raise ValueError(f"Node '{node}' not found in graph '{self.name}'")
        
        target = (node, port)
        if graph_input_name in self.graph_inputs:
            # Fan-out: append new target
            targets = self.graph_inputs[graph_input_name]
            if target not in targets:
                targets.append(target)
        else:
            self.graph_inputs[graph_input_name] = [target]
        return self
    
    def expose_output(
        self,
        graph_output_name: str,
        node: str,
        port: str,
    ) -> ComputeGraph:
        """Объявить выходной порт графа (маппинг на выходной порт узла)."""
        if node not in self.nodes:
            raise ValueError(f"Node '{node}' not found in graph '{self.name}'")
        self.graph_outputs[graph_output_name] = (node, port)
        return self
    
    # ==================== LEGO-ОПЕРАЦИИ ====================
    
    def replace_node(self, name: str, new_block: Any, *, validate: bool = True) -> ComputeGraph:
        """Заменить узел, сохраняя все его соединения.
        
        Это главная Lego-операция: замена одного кирпичика.
        
        Args:
            name: Имя узла для замены.
            new_block: Новый блок.
            validate: Проверять ли совместимость портов нового блока
                      с существующими соединениями (default True).
        """
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' not found in graph '{self.name}'")
        
        if validate:
            self._validate_replacement(name, new_block)
        
        self.nodes[name] = new_block
        return self
    
    def _validate_replacement(self, name: str, new_block: Any):
        """Check that new_block's ports are compatible with existing edges."""
        import logging
        _logger = logging.getLogger(__name__)
        
        new_ports = getattr(new_block, 'declare_io', lambda: {})()
        if not new_ports:
            return  # Can't validate without port declarations
        
        # Check incoming edges (need matching input ports)
        for edge in self.edges:
            if edge.dst_node == name and edge.dst_port not in new_ports:
                _logger.warning(
                    f"Replace warning: new block has no input port '{edge.dst_port}' "
                    f"(required by edge from '{edge.src_node}.{edge.src_port}')"
                )
        
        # Check outgoing edges (need matching output ports)
        for edge in self.edges:
            if edge.src_node == name and edge.src_port not in new_ports:
                _logger.warning(
                    f"Replace warning: new block has no output port '{edge.src_port}' "
                    f"(required by edge to '{edge.dst_node}.{edge.dst_port}')"
                )
    
    def remove_node(self, name: str) -> ComputeGraph:
        """Удалить узел и все его соединения."""
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' not found in graph '{self.name}'")
        del self.nodes[name]
        self.edges = [
            e for e in self.edges
            if e.src_node != name and e.dst_node != name
        ]
        # Remove graph_inputs that reference this node (fan-out aware)
        new_inputs: Dict[str, List[Tuple[str, str]]] = {}
        for k, targets in self.graph_inputs.items():
            filtered = [(n, p) for n, p in targets if n != name]
            if filtered:
                new_inputs[k] = filtered
        self.graph_inputs = new_inputs
        self.graph_outputs = {
            k: v for k, v in self.graph_outputs.items()
            if v[0] != name
        }
        return self
    
    def insert_between(
        self,
        src: str,
        src_port: str,
        dst: str,
        dst_port: str,
        name: str,
        block: Any,
        in_port: str = "input",
        out_port: str = "output",
    ) -> ComputeGraph:
        """Вставить блок между двумя существующими узлами.
        
        Разрывает ребро src.src_port -> dst.dst_port и вставляет:
        src.src_port -> name.in_port, name.out_port -> dst.dst_port
        """
        # Remove the direct edge
        self.edges = [
            e for e in self.edges
            if not (e.src_node == src and e.src_port == src_port
                    and e.dst_node == dst and e.dst_port == dst_port)
        ]
        self.add_node(name, block)
        self.connect(src, src_port, name, in_port)
        self.connect(name, out_port, dst, dst_port)
        return self
    
    def merge_graph(
        self,
        other: ComputeGraph,
        connections: List[Tuple[str, str, str, str]] | None = None,
        prefix: str = "",
    ) -> ComputeGraph:
        """Объединить с другим графом.
        
        Args:
            other: Граф для объединения.
            connections: Список (src, src_port, dst, dst_port) для соединения графов.
            prefix: Префикс для имён узлов из other (для избежания коллизий).
        """
        for node_name, block in other.nodes.items():
            full_name = f"{prefix}{node_name}" if prefix else node_name
            self.add_node(full_name, block)
        
        for edge in other.edges:
            src = f"{prefix}{edge.src_node}" if prefix else edge.src_node
            dst = f"{prefix}{edge.dst_node}" if prefix else edge.dst_node
            self.connect(src, edge.src_port, dst, edge.dst_port)
        
        if connections:
            for src, sp, dst, dp in connections:
                self.connect(src, sp, dst, dp)
        
        return self
    
    def clone(self) -> ComputeGraph:
        """Глубокая копия графа (блоки не копируются — только ссылки)."""
        new_graph = ComputeGraph(self.name)
        new_graph.nodes = OrderedDict(self.nodes)
        new_graph.edges = list(self.edges)
        # Deep-copy fan-out lists
        new_graph.graph_inputs = {k: list(v) for k, v in self.graph_inputs.items()}
        new_graph.graph_outputs = dict(self.graph_outputs)
        new_graph.metadata = dict(self.metadata)
        new_graph._device = self._device
        new_graph._dtype = self._dtype
        return new_graph
    
    # ==================== ВАЛИДАЦИЯ ====================
    
    def validate(self, strict: bool = False) -> List[str]:
        """Validate graph correctness.
        
        Checks:
        1. All edges reference existing nodes
        2. All graph_inputs/graph_outputs reference existing nodes
        3. Graph is acyclic (topological sort succeeds)
        4. Port compatibility on all edges
        5. Undeclared ports on edges (if blocks declare I/O)
        
        Args:
            strict: If True, also raise ValueError on first error.
            
        Returns:
            List of error strings (empty = all good).
        """
        errors = []
        warnings = []
        
        # 1. All edges reference existing nodes
        for edge in self.edges:
            if edge.src_node not in self.nodes:
                errors.append(f"Edge references non-existent source node '{edge.src_node}'")
            if edge.dst_node not in self.nodes:
                errors.append(f"Edge references non-existent destination node '{edge.dst_node}'")
        
        # 2. graph_inputs / graph_outputs reference existing nodes
        for input_name, targets in self.graph_inputs.items():
            for node, port in targets:
                if node not in self.nodes:
                    errors.append(f"Graph input '{input_name}' references non-existent node '{node}'")
        
        for output_name, (node, port) in self.graph_outputs.items():
            if node not in self.nodes:
                errors.append(f"Graph output '{output_name}' references non-existent node '{node}'")
        
        # 3. Acyclicity
        try:
            self.topological_sort()
        except ValueError as e:
            errors.append(str(e))
        
        # 4+5. Port validation
        for edge in self.edges:
            src_block = self.nodes.get(edge.src_node)
            dst_block = self.nodes.get(edge.dst_node)
            if src_block is None or dst_block is None:
                continue
            
            src_ports = getattr(src_block, 'declare_io', lambda: {})()
            dst_ports = getattr(dst_block, 'declare_io', lambda: {})()
            
            # Check undeclared ports
            if src_ports and edge.src_port not in src_ports and edge.src_port != "output":
                warnings.append(
                    f"Edge {edge}: source port '{edge.src_port}' not declared by "
                    f"{getattr(src_block, 'block_type', type(src_block).__name__)}"
                )
            if dst_ports and edge.dst_port not in dst_ports:
                warnings.append(
                    f"Edge {edge}: dest port '{edge.dst_port}' not declared by "
                    f"{getattr(dst_block, 'block_type', type(dst_block).__name__)}"
                )
            
            # Check port compatibility
            src_port = src_ports.get(edge.src_port)
            dst_port = dst_ports.get(edge.dst_port)
            
            if src_port and dst_port:
                try:
                    from yggdrasil.core.block.port import PortValidator
                    valid, msg = PortValidator.validate_connection(
                        getattr(src_block, 'block_type', 'unknown'),
                        src_port,
                        getattr(dst_block, 'block_type', 'unknown'),
                        dst_port,
                    )
                    if not valid:
                        warnings.append(msg)
                except Exception:
                    pass  # PortValidator may not be available
        
        all_issues = errors + warnings
        
        if strict and all_issues:
            raise ValueError(
                f"Graph '{self.name}' validation failed:\n" + 
                "\n".join(f"  - {e}" for e in all_issues)
            )
        
        return all_issues
    
    def topological_sort(self) -> List[str]:
        """Топологическая сортировка узлов (порядок выполнения).
        
        Returns:
            Список имён узлов в порядке выполнения.
        
        Raises:
            ValueError: Если граф содержит цикл.
        """
        # Build adjacency list
        in_degree: Dict[str, int] = {name: 0 for name in self.nodes}
        adj: Dict[str, List[str]] = {name: [] for name in self.nodes}
        
        for edge in self.edges:
            if edge.src_node in adj and edge.dst_node in in_degree:
                adj[edge.src_node].append(edge.dst_node)
                in_degree[edge.dst_node] += 1
        
        # Kahn's algorithm
        queue = deque([n for n, d in in_degree.items() if d == 0])
        result: List[str] = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(self.nodes):
            remaining = set(self.nodes.keys()) - set(result)
            raise ValueError(
                f"Graph '{self.name}' contains a cycle involving nodes: {remaining}"
            )
        
        return result
    
    def get_node_dependencies(self, node_name: str) -> Set[str]:
        """Получить все узлы, от которых зависит данный узел (рекурсивно)."""
        deps: Set[str] = set()
        queue = deque([node_name])
        
        while queue:
            current = queue.popleft()
            for edge in self.edges:
                if edge.dst_node == current and edge.src_node not in deps:
                    deps.add(edge.src_node)
                    queue.append(edge.src_node)
        
        return deps
    
    # ==================== СЕРИАЛИЗАЦИЯ ====================
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> ComputeGraph:
        """Загрузить граф из YAML-файла.
        
        Формат::
        
            name: my_pipeline
            nodes:
              clip:
                block: conditioner/clip_text
                config: {pretrained: openai/clip-vit-large-patch14}
              unet:
                block: backbone/unet2d_condition
                config: {pretrained: stable-diffusion-v1-5}
            edges:
              - [clip.embedding, unet.condition]
            inputs:
              prompt: [clip, text]
            outputs:
              result: [unet, output]
        """
        from yggdrasil.core.block.builder import BlockBuilder
        
        conf = OmegaConf.load(path)
        graph = cls(conf.get("name", "unnamed"))
        graph.metadata = dict(conf.get("metadata", {}))
        
        # Build nodes
        for node_name, node_conf in conf.get("nodes", {}).items():
            block_type = node_conf.get("block") or node_conf.get("type")
            config = dict(node_conf.get("config", {}))
            config["type"] = block_type
            block = BlockBuilder.build(config)
            graph.add_node(node_name, block)
        
        # Build edges: "src.port -> dst.port" or [src.port, dst.port]
        for edge_def in conf.get("edges", []):
            if isinstance(edge_def, str):
                pass  # fall through to string parsing below
            elif hasattr(edge_def, '__getitem__') and len(edge_def) >= 2:
                src_spec, dst_spec = str(edge_def[0]), str(edge_def[1])
                src_node, src_port = src_spec.split(".", 1)
                dst_node, dst_port = dst_spec.split(".", 1)
                graph.connect(src_node.strip(), src_port.strip(), dst_node.strip(), dst_port.strip())
                continue
            else:
                # String format: "src.port -> dst.port"
                parts = str(edge_def).split("->")
                src_spec = parts[0].strip()
                dst_spec = parts[1].strip()
            
            src_node, src_port = str(src_spec).split(".", 1)
            dst_node, dst_port = str(dst_spec).split(".", 1)
            graph.connect(src_node.strip(), src_port.strip(), dst_node.strip(), dst_port.strip())
        
        # Graph inputs (supports fan-out: list of [node, port] pairs)
        for input_name, mapping in conf.get("inputs", {}).items():
            if isinstance(mapping, str):
                node, port = mapping.split(".", 1)
                graph.expose_input(input_name, node.strip(), port.strip())
            elif hasattr(mapping, '__getitem__') and len(mapping) >= 2:
                # Check if it's a list of pairs (fan-out) or a single pair
                first = mapping[0]
                if hasattr(first, '__getitem__') and not isinstance(first, str) and len(first) >= 2:
                    # Fan-out: [[node, port], [node, port], ...]
                    for pair in mapping:
                        graph.expose_input(input_name, str(pair[0]), str(pair[1]))
                else:
                    # Single pair: [node, port]
                    graph.expose_input(input_name, str(mapping[0]), str(mapping[1]))
            else:
                node, port = str(mapping).split(".", 1)
                graph.expose_input(input_name, node.strip(), port.strip())
        
        # Graph outputs
        for output_name, mapping in conf.get("outputs", {}).items():
            if isinstance(mapping, str):
                node, port = mapping.split(".", 1)
                graph.expose_output(output_name, node.strip(), port.strip())
            elif hasattr(mapping, '__getitem__') and len(mapping) >= 2:
                graph.expose_output(output_name, str(mapping[0]), str(mapping[1]))
            else:
                node, port = str(mapping).split(".", 1)
                graph.expose_output(output_name, node.strip(), port.strip())
        
        return graph
    
    def to_yaml(self, path: str | Path) -> None:
        """Сохранить граф в YAML-файл."""
        data = {
            "name": self.name,
            "metadata": self.metadata,
            "nodes": {},
            "edges": [],
            "inputs": {},
            "outputs": {},
        }
        
        for node_name, block in self.nodes.items():
            block_type = getattr(block, 'block_type', 'unknown')
            config = {}
            if hasattr(block, 'config'):
                try:
                    config = OmegaConf.to_container(block.config, resolve=True)
                except Exception:
                    config = dict(block.config) if block.config else {}
            data["nodes"][node_name] = {
                "block": block_type,
                "config": config,
            }
        
        for edge in self.edges:
            data["edges"].append([
                f"{edge.src_node}.{edge.src_port}",
                f"{edge.dst_node}.{edge.dst_port}",
            ])
        
        for input_name, targets in self.graph_inputs.items():
            if len(targets) == 1:
                data["inputs"][input_name] = [targets[0][0], targets[0][1]]
            else:
                data["inputs"][input_name] = [[n, p] for n, p in targets]
        
        for output_name, (node, port) in self.graph_outputs.items():
            data["outputs"][output_name] = [node, port]
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(data), path)
    
    # ==================== WORKFLOW SERIALIZATION ====================
    
    def to_workflow(self, path: str | Path, parameters: dict | None = None) -> None:
        """Save a complete workflow: graph structure + runtime parameters.
        
        This is the ComfyUI-like feature: save everything needed to reproduce
        a generation, then replay it with `from_workflow()`.
        
        Args:
            path: Destination file (.yaml or .json)
            parameters: Runtime parameters to include (prompt, seed, guidance_scale, etc.)
        
        Format::
        
            name: sd15_txt2img
            metadata: {...}
            nodes: {...}
            edges: [...]
            inputs: {...}
            outputs: {...}
            parameters:
              prompt: {text: "a beautiful cat"}
              guidance_scale: 7.5
              seed: 42
              num_steps: 28
        """
        import json
        
        # Build base graph structure (same as to_yaml)
        data = {
            "name": self.name,
            "metadata": dict(self.metadata),
            "nodes": {},
            "edges": [],
            "inputs": {},
            "outputs": {},
            "parameters": parameters or {},
        }
        
        for node_name, block in self.nodes.items():
            block_type = getattr(block, 'block_type', 'unknown')
            config = {}
            if hasattr(block, 'config'):
                try:
                    config = OmegaConf.to_container(block.config, resolve=True)
                except Exception:
                    config = dict(block.config) if block.config else {}
            data["nodes"][node_name] = {
                "block": block_type,
                "config": config,
            }
        
        for edge in self.edges:
            data["edges"].append([
                f"{edge.src_node}.{edge.src_port}",
                f"{edge.dst_node}.{edge.dst_port}",
            ])
        
        for input_name, targets in self.graph_inputs.items():
            if len(targets) == 1:
                data["inputs"][input_name] = [targets[0][0], targets[0][1]]
            else:
                data["inputs"][input_name] = [[n, p] for n, p in targets]
        
        for output_name, (node, port) in self.graph_outputs.items():
            data["outputs"][output_name] = [node, port]
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        suffix = path.suffix.lower()
        if suffix == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            OmegaConf.save(OmegaConf.create(data), path)
    
    @classmethod
    def from_workflow(cls, path: str | Path) -> tuple[ComputeGraph, dict]:
        """Load a complete workflow: reconstructs graph + returns runtime parameters.
        
        Args:
            path: Workflow file (.yaml or .json)
            
        Returns:
            Tuple of (graph, parameters) where parameters is a dict of
            runtime inputs (prompt, seed, etc.) that were saved with the workflow.
        
        Example::
        
            graph, params = ComputeGraph.from_workflow("workflow.yaml")
            pipe = Pipeline.from_graph(graph)
            output = pipe(**params)
        """
        import json
        
        path = Path(path)
        suffix = path.suffix.lower()
        
        if suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        else:
            data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        
        # Build graph from structure using from_yaml-like logic
        graph = cls._build_from_dict(data)
        
        # Extract parameters
        parameters = data.get("parameters", {})
        
        return graph, parameters
    
    @classmethod
    def _build_from_dict(cls, data: dict) -> ComputeGraph:
        """Build a ComputeGraph from a dict structure."""
        from yggdrasil.core.block.builder import BlockBuilder
        
        graph = cls(data.get("name", "unnamed"))
        graph.metadata = dict(data.get("metadata", {}))
        
        # Build nodes
        for node_name, node_conf in data.get("nodes", {}).items():
            block_type = node_conf.get("block") or node_conf.get("type")
            config = dict(node_conf.get("config", {}))
            config["type"] = block_type
            block = BlockBuilder.build(config)
            graph.add_node(node_name, block)
        
        # Build edges
        for edge_def in data.get("edges", []):
            if isinstance(edge_def, str):
                parts = edge_def.split("->")
                src_spec, dst_spec = parts[0].strip(), parts[1].strip()
            elif isinstance(edge_def, (list, tuple)) and len(edge_def) >= 2:
                src_spec, dst_spec = str(edge_def[0]), str(edge_def[1])
            else:
                continue
            
            src_node, src_port = src_spec.split(".", 1)
            dst_node, dst_port = dst_spec.split(".", 1)
            graph.connect(src_node.strip(), src_port.strip(), 
                         dst_node.strip(), dst_port.strip())
        
        # Graph inputs
        for input_name, mapping in data.get("inputs", {}).items():
            if isinstance(mapping, str):
                node, port = mapping.split(".", 1)
                graph.expose_input(input_name, node.strip(), port.strip())
            elif isinstance(mapping, (list, tuple)):
                first = mapping[0]
                if isinstance(first, (list, tuple)):
                    for pair in mapping:
                        graph.expose_input(input_name, str(pair[0]), str(pair[1]))
                else:
                    graph.expose_input(input_name, str(mapping[0]), str(mapping[1]))
        
        # Graph outputs
        for output_name, mapping in data.get("outputs", {}).items():
            if isinstance(mapping, str):
                node, port = mapping.split(".", 1)
                graph.expose_output(output_name, node.strip(), port.strip())
            elif isinstance(mapping, (list, tuple)):
                graph.expose_output(output_name, str(mapping[0]), str(mapping[1]))
        
        return graph
    
    # ==================== ВИЗУАЛИЗАЦИЯ ====================
    
    def visualize(self) -> str:
        """Сгенерировать Mermaid-диаграмму графа."""
        lines = ["graph LR"]
        
        # Nodes
        for name, block in self.nodes.items():
            block_type = getattr(block, 'block_type', 'unknown')
            label = f'{name}["{name}\\n{block_type}"]'
            lines.append(f"    {label}")
        
        # Graph inputs (fan-out aware)
        for input_name, targets in self.graph_inputs.items():
            safe_id = f"in_{input_name}"
            lines.append(f"    {safe_id}(({input_name}))")
            for node, port in targets:
                lines.append(f"    {safe_id} -->|{port}| {node}")
        
        # Edges
        for edge in self.edges:
            label = f"{edge.src_port} -> {edge.dst_port}"
            lines.append(f'    {edge.src_node} -->|"{label}"| {edge.dst_node}')
        
        # Graph outputs
        for output_name, (node, port) in self.graph_outputs.items():
            safe_id = f"out_{output_name}"
            lines.append(f"    {safe_id}(({output_name}))")
            lines.append(f"    {node} -->|{port}| {safe_id}")
        
        return "\n".join(lines)
    
    # ==================== QUERY ====================
    
    def get_edges_from(self, node_name: str) -> List[Edge]:
        """Получить все исходящие рёбра от узла."""
        return [e for e in self.edges if e.src_node == node_name]
    
    def get_edges_to(self, node_name: str) -> List[Edge]:
        """Получить все входящие рёбра к узлу."""
        return [e for e in self.edges if e.dst_node == node_name]
    
    def get_connected_inputs(self, node_name: str) -> Set[str]:
        """Получить имена входных портов узла, к которым подключены рёбра."""
        return {e.dst_port for e in self.edges if e.dst_node == node_name}
    
    def list_nodes(self) -> List[str]:
        """Список имён всех узлов."""
        return list(self.nodes.keys())
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def __contains__(self, node_name: str) -> bool:
        return node_name in self.nodes
    
    def __getitem__(self, node_name: str) -> Any:
        return self.nodes[node_name]
    
    def __repr__(self) -> str:
        total_targets = sum(len(t) for t in self.graph_inputs.values())
        return (
            f"<ComputeGraph '{self.name}' "
            f"nodes={len(self.nodes)} edges={len(self.edges)} "
            f"inputs={list(self.graph_inputs.keys())}({total_targets} targets) "
            f"outputs={list(self.graph_outputs.keys())}>"
        )
    
    # ==================== CLASS METHODS ====================
    
    @classmethod
    def from_template(
        cls,
        template_name: str,
        *,
        device: Any = None,
        dtype: Any = None,
        **kwargs,
    ) -> ComputeGraph:
        """Создать граф из именованного шаблона.
        
        Args:
            template_name: Имя шаблона ("sd15_txt2img", "flux_txt2img", ...).
            device: Устройство ("cuda", "mps", "cpu"). Если указано,
                    граф сразу переносится на устройство.
            dtype: Тип данных. Если None — выбирается автоматически.
            **kwargs: Доп. параметры шаблона (pretrained, ...).
        
        Returns:
            Готовый ComputeGraph (уже на устройстве, если указан device).
        
        Пример::
        
            graph = ComputeGraph.from_template("sd15_txt2img", device="cuda")
            outputs = graph.execute(prompt="a cat", num_steps=28)
        """
        from yggdrasil.core.graph.templates import get_template
        builder_fn = get_template(template_name)
        graph = builder_fn(**kwargs)
        if device is not None:
            graph.to(device, dtype)
        return graph
    
    def execute(self, *, prompt=None, **kwargs: Any) -> Dict[str, Any]:
        """Execute the graph.
        
        Two modes of operation:
        
        1. **High-level** (convenience — delegates to Pipeline):
           Accepts prompt, guidance_scale, num_steps, seed, width, height, etc.
           Auto-prepares noise latents and applies overrides.
        
        2. **Low-level** (raw graph execution):
           Pass ready-made graph inputs directly.
        
        Example::
        
            # High-level (auto noise, auto guidance)
            outputs = graph.execute(prompt="a cat", guidance_scale=7.5, num_steps=28, seed=42)
            
            # Low-level (manual)
            outputs = graph.execute(latents=my_noise, prompt={"text": "a cat"})
        
        Returns:
            Dict with graph outputs (keys = expose_output names).
        """
        from yggdrasil.pipeline import Pipeline
        pipe = Pipeline.from_graph(self)
        
        # Extract high-level params if present
        guidance_scale = kwargs.pop("guidance_scale", None)
        num_steps = kwargs.pop("num_steps", None)
        width = kwargs.pop("width", None)
        height = kwargs.pop("height", None)
        seed = kwargs.pop("seed", None)
        batch_size = kwargs.pop("batch_size", 1)
        negative_prompt = kwargs.pop("negative_prompt", None)
        
        result = pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            width=width,
            height=height,
            seed=seed,
            batch_size=batch_size,
            negative_prompt=negative_prompt,
            **kwargs,
        )
        return result.raw
    
    def execute_raw(self, **inputs: Any) -> Dict[str, Any]:
        """Execute graph with raw inputs (no Pipeline convenience).
        
        Use this for non-diffusion graphs or when you want full control.
        """
        from .executor import GraphExecutor
        return GraphExecutor().execute(self, **inputs)
    
    def _iter_all_blocks(self):
        """Итерация по ВСЕМ блокам, включая вложенные SubGraph."""
        for name, block in self.nodes.items():
            yield name, block
            if hasattr(block, 'graph') and block.graph is not None:
                for inner_name, inner_block in block.graph.nodes.items():
                    yield f"{name}.{inner_name}", inner_block
