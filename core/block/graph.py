from typing import Dict, List
import networkx as nx
from .base import AbstractBlock


class BlockGraph:
    """Визуализация и анализ графа блоков."""
    
    @staticmethod
    def build_from_block(root: AbstractBlock) -> nx.DiGraph:
        """Строит граф зависимостей."""
        G = nx.DiGraph()
        
        def add_node(block: AbstractBlock):
            G.add_node(block.block_id, type=block.block_type)
            for slot_name, child in block.children.items():
                if isinstance(child, list):
                    for c in child:
                        G.add_edge(block.block_id, c.block_id, slot=slot_name)
                        add_node(c)
                else:
                    G.add_edge(block.block_id, child.block_id, slot=slot_name)
                    add_node(child)
        
        add_node(root)
        return G
    
    @staticmethod
    def to_mermaid(root: AbstractBlock) -> str:
        """Генерирует Mermaid-диаграмму (можно вставить в docs)."""
        G = BlockGraph.build_from_block(root)
        lines = ["graph TD"]
        
        for node, data in G.nodes(data=True):
            lines.append(f'    {node}["{data["type"]}"]')
        
        for u, v, data in G.edges(data=True):
            lines.append(f'    {u} -->|{data.get("slot", "")}| {v}')
        
        return "\n".join(lines)