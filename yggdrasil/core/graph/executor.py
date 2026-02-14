# yggdrasil/core/graph/executor.py
"""GraphExecutor — исполнитель вычислительного графа.

Выполняет блоки в топологическом порядке, передавая данные через порты.
"""
from __future__ import annotations

import time
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import torch

from .graph import ComputeGraph, Edge

logger = logging.getLogger(__name__)


class GraphExecutor:
    """Исполнитель вычислительного графа.
    
    Выполняет узлы в топологическом порядке:
    1. Собирает входы для каждого узла из кэша выходов предыдущих узлов
    2. Вызывает block.process(**inputs)
    3. Кэширует выходы
    4. Возвращает выходы графа
    
    Поддерживает:
    - Кэширование промежуточных результатов
    - Callbacks для мониторинга
    - torch.no_grad() для инференса
    - Debug-режим с логированием
    """
    
    def __init__(
        self,
        debug: bool = False,
        no_grad: bool = True,
        callbacks: List[Callable] | None = None,
    ):
        self.debug = debug
        self.no_grad = no_grad
        self.callbacks = callbacks or []
    
    def execute(
        self,
        graph: ComputeGraph,
        **inputs: Any,
    ) -> Dict[str, Any]:
        """Выполнить граф.
        
        Args:
            graph: Вычислительный граф.
            **inputs: Именованные входы (ключи = graph_input_names).
        
        Returns:
            Dict с именованными выходами (ключи = graph_output_names).
        """
        if self.no_grad:
            with torch.no_grad():
                return self._execute_impl(graph, inputs)
        else:
            return self._execute_impl(graph, inputs)
    
    def execute_training(
        self,
        graph: ComputeGraph,
        **inputs: Any,
    ) -> Dict[str, Any]:
        """Выполнить граф в режиме обучения (с градиентами)."""
        return self._execute_impl(graph, inputs)
    
    def _execute_impl(
        self,
        graph: ComputeGraph,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Внутренняя реализация выполнения."""
        order = graph.topological_sort()
        cache: Dict[str, Dict[str, Any]] = {}  # node_name -> {port_name: value}
        
        if self.debug:
            logger.info(f"Executing graph '{graph.name}' with {len(order)} nodes")
            logger.info(f"Execution order: {order}")
            logger.info(f"Graph inputs: {list(inputs.keys())}")
        
        for node_name in order:
            block = graph.nodes[node_name]
            
            # 1. Gather inputs for this node
            port_inputs = self._gather_inputs(node_name, graph, cache, inputs)
            
            if self.debug:
                input_keys = list(port_inputs.keys())
                logger.info(f"  [{node_name}] inputs: {input_keys}")
            
            # 2. Check required ports
            self._check_required_ports(node_name, block, port_inputs)
            
            # 3. Execute block
            t0 = time.time()
            try:
                port_outputs = self._execute_node(block, port_inputs)
            except Exception as e:
                raise RuntimeError(
                    f"Error executing node '{node_name}' "
                    f"(block_type={getattr(block, 'block_type', 'unknown')}): {e}"
                ) from e
            
            elapsed = time.time() - t0
            
            if self.debug:
                output_keys = list(port_outputs.keys())
                logger.info(f"  [{node_name}] outputs: {output_keys} ({elapsed:.3f}s)")
            
            # 3. Cache outputs
            cache[node_name] = port_outputs
            
            # 4. Callbacks
            for cb in self.callbacks:
                cb(node_name, port_inputs, port_outputs, elapsed)
        
        # 5. Gather graph outputs
        return self._gather_outputs(graph, cache)
    
    def _execute_node(
        self,
        block: Any,
        port_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Выполнить один узел.
        
        Пытается вызвать block.process(**port_inputs).
        Если process не переопределён, использует forward().
        """
        # Try process() first (port-based API)
        if hasattr(block, 'process'):
            result = block.process(**port_inputs)
            if isinstance(result, dict):
                return result
            return {"output": result}
        
        # Fallback to forward()
        if hasattr(block, 'forward'):
            result = block.forward(**port_inputs)
            if isinstance(result, dict):
                return result
            return {"output": result}
        
        # Callable blocks
        if callable(block):
            result = block(**port_inputs)
            if isinstance(result, dict):
                return result
            return {"output": result}
        
        raise TypeError(
            f"Block {type(block).__name__} is not executable "
            f"(no process(), forward(), or __call__ method)"
        )
    
    def _gather_inputs(
        self,
        node_name: str,
        graph: ComputeGraph,
        cache: Dict[str, Dict[str, Any]],
        graph_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Собрать входы для узла из кэша и graph_inputs."""
        port_inputs: Dict[str, Any] = {}
        
        # 1. From graph_inputs (exposed inputs) — fan-out aware
        for input_name, targets in graph.graph_inputs.items():
            if input_name not in graph_inputs:
                continue
            for target_node, target_port in targets:
                if target_node == node_name:
                    port_inputs[target_port] = graph_inputs[input_name]
        
        # 2. From edges (outputs of other nodes)
        for edge in graph.edges:
            if edge.dst_node != node_name:
                continue
            
            src_cache = cache.get(edge.src_node, {})
            if edge.src_port in src_cache:
                value = src_cache[edge.src_port]
            else:
                # Try "output" as default port name
                value = src_cache.get("output")
            
            # Check if this input port already has a value (multiple connections)
            if edge.dst_port in port_inputs:
                # Convert to list for multiple inputs
                existing = port_inputs[edge.dst_port]
                if not isinstance(existing, list):
                    port_inputs[edge.dst_port] = [existing, value]
                else:
                    port_inputs[edge.dst_port].append(value)
            else:
                port_inputs[edge.dst_port] = value
        
        return port_inputs
    
    def _check_required_ports(
        self,
        node_name: str,
        block: Any,
        port_inputs: Dict[str, Any],
    ):
        """Warn if required input ports are missing (debug mode only)."""
        if not self.debug:
            return
        
        ports = getattr(block, 'declare_io', lambda: {})()
        if not ports:
            return
        
        for port_name, port in ports.items():
            if port.direction == "input" and not port.optional:
                if port_name not in port_inputs or port_inputs[port_name] is None:
                    logger.warning(
                        f"Node '{node_name}': required input port '{port_name}' "
                        f"is not connected or is None"
                    )
    
    def _gather_outputs(
        self,
        graph: ComputeGraph,
        cache: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Собрать выходы графа из кэша."""
        outputs = {}
        
        for output_name, (node, port) in graph.graph_outputs.items():
            node_cache = cache.get(node, {})
            if port in node_cache:
                outputs[output_name] = node_cache[port]
            elif "output" in node_cache:
                outputs[output_name] = node_cache["output"]
            else:
                outputs[output_name] = None
        
        # If no explicit outputs defined, return all cached outputs from last node
        if not outputs and cache:
            last_node = list(cache.keys())[-1]
            outputs = cache[last_node]
        
        return outputs
