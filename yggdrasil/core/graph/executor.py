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
    """Graph executor — runs blocks in topological order via ports.
    
    Execution model:
    1. Topological sort to determine order
    2. For each node: gather inputs from cache + graph_inputs
    3. Validate required ports (strict by default)
    4. Call block.process(**inputs)
    5. Cache outputs
    6. Return graph outputs
    
    Supports:
    - Strict port validation (default: True — fail on missing required ports)
    - Callbacks for monitoring (node_name, inputs, outputs, elapsed)
    - torch.no_grad() for inference
    - Debug mode with detailed logging
    """
    
    def __init__(
        self,
        debug: bool = False,
        no_grad: bool = True,
        callbacks: List[Callable] | None = None,
        strict: bool = True,
        enable_cache: bool = False,
    ):
        self.debug = debug
        self.no_grad = no_grad
        self.callbacks = callbacks or []
        self.strict = strict  # Fail on missing required ports (default True)
        self.enable_cache = enable_cache
        self._cache: Dict[str, Dict[str, Any]] = {}  # node_name -> outputs
        self._cache_keys: Dict[str, str] = {}  # node_name -> input_hash
        self._invalidated: set = set()  # nodes to re-execute
    
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
        """Internal execution implementation."""
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
            
            # Smart caching: skip re-execution if inputs unchanged
            if self.enable_cache and node_name not in self._invalidated:
                input_key = self._compute_input_hash(port_inputs)
                if node_name in self._cache_keys and self._cache_keys[node_name] == input_key:
                    cache[node_name] = self._cache[node_name]
                    if self.debug:
                        logger.info(f"  [{node_name}] CACHED (skipped)")
                    continue
            
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
            
            # 4. Cache outputs
            cache[node_name] = port_outputs
            
            # Update smart cache
            if self.enable_cache:
                input_key = self._compute_input_hash(port_inputs)
                self._cache[node_name] = port_outputs
                self._cache_keys[node_name] = input_key
            
            # 5. Callbacks
            for cb in self.callbacks:
                cb(node_name, port_inputs, port_outputs, elapsed)
        
        # Clear invalidation set after execution
        self._invalidated.clear()
        
        # 6. Gather graph outputs
        return self._gather_outputs(graph, cache)
    
    def invalidate(self, node_name: str):
        """Force re-computation of a node on next execution.
        
        Also invalidates all downstream nodes automatically.
        """
        self._invalidated.add(node_name)
        # Also invalidate any cached entry
        self._cache.pop(node_name, None)
        self._cache_keys.pop(node_name, None)
    
    def clear_cache(self):
        """Clear all cached node outputs."""
        self._cache.clear()
        self._cache_keys.clear()
        self._invalidated.clear()
    
    @staticmethod
    def _compute_input_hash(port_inputs: Dict[str, Any]) -> str:
        """Compute a hash key for node inputs to detect changes.
        
        Uses id() for tensors (fast, tracks exact object identity) and
        repr() for other types.
        """
        import hashlib
        parts = []
        for key in sorted(port_inputs.keys()):
            val = port_inputs[key]
            if isinstance(val, torch.Tensor):
                # For tensors: use shape + data_ptr for identity
                parts.append(f"{key}:tensor:{val.shape}:{val.data_ptr()}")
            elif val is None:
                parts.append(f"{key}:None")
            else:
                parts.append(f"{key}:{id(val)}")
        return hashlib.md5("|".join(parts).encode()).hexdigest()
    
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
        """Check required input ports.
        
        In strict mode (default): raises ValueError for missing required ports.
        In non-strict mode: always logs a warning (not just in debug).
        """
        ports = getattr(block, 'declare_io', lambda: {})()
        if not ports:
            return
        
        for port_name, port in ports.items():
            if port.direction == "input" and not getattr(port, 'optional', False):
                if port_name not in port_inputs or port_inputs[port_name] is None:
                    block_type = getattr(block, 'block_type', type(block).__name__)
                    msg = (
                        f"Node '{node_name}' ({block_type}): "
                        f"required input port '{port_name}' is not connected or is None"
                    )
                    if self.strict:
                        raise ValueError(msg)
                    else:
                        logger.warning(msg)
    
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
