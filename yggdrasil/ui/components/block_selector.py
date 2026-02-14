"""Block selector component for Gradio UI."""
from __future__ import annotations
from typing import Dict, List, Any


def get_available_blocks() -> Dict[str, List[str]]:
    """Get all registered blocks grouped by category."""
    from yggdrasil.core.block.registry import BlockRegistry
    
    blocks = BlockRegistry.list_blocks()
    categorized = {}
    
    for key in sorted(blocks.keys()):
        category = key.split("/")[0] if "/" in key else "other"
        if category not in categorized:
            categorized[category] = []
        categorized[category].append(key)
    
    return categorized


def get_available_plugins() -> List[Dict[str, str]]:
    """Get all registered plugins with metadata."""
    from yggdrasil.plugins.base import PluginRegistry
    
    plugins = []
    for name, cls in PluginRegistry.list_plugins().items():
        plugins.append({
            "name": name,
            "modality": getattr(cls, "modality", name),
            "description": getattr(cls, "description", ""),
            "version": getattr(cls, "version", "1.0.0"),
        })
    
    return plugins
