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
        """Validate port compatibility for an edge (warning-level, not blocking)."""
        import logging
        _logger = logging.getLogger(__name__)
        
        src_block = self.nodes[src]
        dst_block = self.nodes[dst]
        
        src_ports = getattr(src_block, 'declare_io', lambda: {})()
        dst_ports = getattr(dst_block, 'declare_io', lambda: {})()
        
        if not src_ports or not dst_ports:
            return  # Can't validate without port declarations
        
        sp = src_ports.get(src_port)
        dp = dst_ports.get(dst_port)
        
        if sp is not None and dp is not None:
            from yggdrasil.core.block.port import PortValidator
            valid, msg = PortValidator.validate_connection(
                getattr(src_block, 'block_type', 'unknown'), sp,
                getattr(dst_block, 'block_type', 'unknown'), dp,
            )
            if not valid:
                _logger.warning(f"Port compatibility warning: {msg}")
    
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
        return new_graph
    
    # ==================== ВАЛИДАЦИЯ ====================
    
    def validate(self) -> List[str]:
        """Проверить корректность графа.
        
        Returns:
            Список ошибок (пустой = всё ок).
        """
        errors = []
        
        # 1. Проверяем, что все рёбра ссылаются на существующие узлы
        for edge in self.edges:
            if edge.src_node not in self.nodes:
                errors.append(f"Edge references non-existent source node '{edge.src_node}'")
            if edge.dst_node not in self.nodes:
                errors.append(f"Edge references non-existent destination node '{edge.dst_node}'")
        
        # 2. Проверяем graph_inputs / graph_outputs
        for input_name, targets in self.graph_inputs.items():
            for node, port in targets:
                if node not in self.nodes:
                    errors.append(f"Graph input '{input_name}' references non-existent node '{node}'")
        
        for output_name, (node, port) in self.graph_outputs.items():
            if node not in self.nodes:
                errors.append(f"Graph output '{output_name}' references non-existent node '{node}'")
        
        # 3. Проверяем ацикличность
        try:
            self.topological_sort()
        except ValueError as e:
            errors.append(str(e))
        
        # 4. Проверяем порты (если блоки имеют declare_io)
        from yggdrasil.core.block.port import PortValidator
        
        for edge in self.edges:
            src_block = self.nodes.get(edge.src_node)
            dst_block = self.nodes.get(edge.dst_node)
            if src_block is None or dst_block is None:
                continue
            
            src_ports = getattr(src_block, 'declare_io', lambda: {})()
            dst_ports = getattr(dst_block, 'declare_io', lambda: {})()
            
            src_port = src_ports.get(edge.src_port)
            dst_port = dst_ports.get(edge.dst_port)
            
            if src_port and dst_port:
                valid, msg = PortValidator.validate_connection(
                    getattr(src_block, 'block_type', 'unknown'),
                    src_port,
                    getattr(dst_block, 'block_type', 'unknown'),
                    dst_port,
                )
                if not valid:
                    errors.append(msg)
        
        return errors
    
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
            if isinstance(edge_def, (list, tuple)):
                src_spec, dst_spec = edge_def[0], edge_def[1]
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
            if isinstance(mapping, (list, tuple)):
                # Check if it's a list of pairs (fan-out) or a single pair
                if len(mapping) > 0 and isinstance(mapping[0], (list, tuple)):
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
            if isinstance(mapping, (list, tuple)):
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
    def from_template(cls, template_name: str, **kwargs) -> ComputeGraph:
        """Создать граф из именованного шаблона.
        
        Args:
            template_name: Имя шаблона (например "sd15_txt2img", "flux_txt2img").
            **kwargs: Параметры для шаблона.
        
        Returns:
            Готовый ComputeGraph.
        """
        from yggdrasil.core.graph.templates import get_template
        builder_fn = get_template(template_name)
        return builder_fn(**kwargs)
    
    def execute(self, **inputs: Any) -> Dict[str, Any]:
        """Выполнить граф с данными входами.
        
        Shortcut для ``GraphExecutor().execute(self, **inputs)``.
        """
        from .executor import GraphExecutor
        return GraphExecutor().execute(self, **inputs)
