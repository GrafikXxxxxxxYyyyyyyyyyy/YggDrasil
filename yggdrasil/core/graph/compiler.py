# yggdrasil/core/graph/compiler.py
"""Graph Compiler — оптимизация и компиляция графа.

Компилятор анализирует граф и применяет оптимизации:
- Фьюзинг последовательных узлов
- Удаление неиспользуемых узлов (dead code elimination)
- Вычисление статических подграфов
- Подготовка к torch.compile()
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

from .graph import ComputeGraph

logger = logging.getLogger(__name__)


class GraphCompiler:
    """Компилятор и оптимизатор графа.
    
    Применяет серию проходов для оптимизации графа перед выполнением.
    """
    
    def __init__(self, passes: List[str] | None = None):
        """
        Args:
            passes: Список оптимизаций для применения.
                    По умолчанию все: ["dce", "validate", "order"].
        """
        self.passes = passes or ["dce", "validate", "order"]
    
    def compile(self, graph: ComputeGraph) -> ComputeGraph:
        """Скомпилировать граф (применить оптимизации).
        
        Возвращает НОВЫЙ оптимизированный граф (оригинал не меняется).
        """
        result = graph.clone()
        
        for pass_name in self.passes:
            if pass_name == "dce":
                result = self._dead_code_elimination(result)
            elif pass_name == "validate":
                errors = result.validate()
                if errors:
                    raise ValueError(
                        f"Graph validation failed:\n" +
                        "\n".join(f"  - {e}" for e in errors)
                    )
            elif pass_name == "order":
                # Just verify topological sort works
                result.topological_sort()
            else:
                logger.warning(f"Unknown compiler pass: {pass_name}")
        
        return result
    
    def _dead_code_elimination(self, graph: ComputeGraph) -> ComputeGraph:
        """Удалить узлы, выходы которых не используются.
        
        Работает от выходов графа назад: если узел не достижим
        от выходных узлов, он удаляется.
        """
        # Find all nodes that contribute to graph outputs
        output_nodes = {node for node, _ in graph.graph_outputs.values()}
        reachable = set(output_nodes)
        
        # BFS backwards
        queue = list(output_nodes)
        while queue:
            node = queue.pop(0)
            for edge in graph.edges:
                if edge.dst_node == node and edge.src_node not in reachable:
                    reachable.add(edge.src_node)
                    queue.append(edge.src_node)
        
        # Also keep nodes that are targets of graph inputs (fan-out aware)
        for _, targets in graph.graph_inputs.items():
            for node, _ in targets:
                reachable.add(node)
                # And everything reachable from input nodes
                queue = [node]
                while queue:
                    n = queue.pop(0)
                    for edge in graph.edges:
                        if edge.src_node == n and edge.dst_node not in reachable:
                            reachable.add(edge.dst_node)
                            queue.append(edge.dst_node)
        
        # Remove unreachable nodes
        unreachable = set(graph.nodes.keys()) - reachable
        if unreachable:
            logger.info(f"DCE: removing {len(unreachable)} unreachable nodes: {unreachable}")
            for node_name in unreachable:
                graph.remove_node(node_name)
        
        return graph
    
    @staticmethod
    def estimate_memory(graph: ComputeGraph) -> Dict[str, Any]:
        """Оценить потребление памяти графа.
        
        Returns:
            Dict с оценками для каждого узла и общей суммой.
        """
        import torch
        
        estimates = {}
        total_params = 0
        
        for name, block in graph.nodes.items():
            if hasattr(block, 'parameters'):
                n_params = sum(p.numel() for p in block.parameters())
                mem_bytes = sum(
                    p.numel() * p.element_size() for p in block.parameters()
                )
            else:
                n_params = 0
                mem_bytes = 0
            
            estimates[name] = {
                "params": n_params,
                "memory_mb": mem_bytes / (1024 * 1024),
                "block_type": getattr(block, 'block_type', 'unknown'),
            }
            total_params += n_params
        
        return {
            "nodes": estimates,
            "total_params": total_params,
            "total_memory_mb": sum(e["memory_mb"] for e in estimates.values()),
        }
