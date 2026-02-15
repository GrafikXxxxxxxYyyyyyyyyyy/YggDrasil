"""Graph visualizer -- generate Mermaid/Graphviz diagrams of model graphs.

Usage:
    python -m yggdrasil.tools.graph_visualizer --config configs/recipes/sd15_generate.yaml
"""
from __future__ import annotations

import argparse
from typing import Optional, Dict, Any


def model_to_mermaid(model, title: str = "Model Graph") -> str:
    """Generate Mermaid diagram from a ModularDiffusionModel.
    
    Args:
        model: ModularDiffusionModel instance
        title: Diagram title
        
    Returns:
        Mermaid diagram string
    """
    lines = [f"graph TB"]
    lines.append(f"    subgraph {_safe_id(title)} [{title}]")
    
    model_id = _safe_id(getattr(model, "block_id", "model"))
    model_type = getattr(model, "block_type", "model")
    lines.append(f"    {model_id}[{model_type}]")
    
    _add_children(lines, model, model_id, depth=1)
    
    lines.append("    end")
    return "\n".join(lines)


def _add_children(lines: list, block, parent_id: str, depth: int):
    """Recursively add child blocks to the diagram."""
    children = getattr(block, "_slot_children", {})
    
    for slot_name, child in children.items():
        if child is None:
            continue
        
        if isinstance(child, list):
            for i, c in enumerate(child):
                child_id = _safe_id(f"{parent_id}_{slot_name}_{i}")
                child_type = getattr(c, "block_type", type(c).__name__)
                indent = "    " * (depth + 1)
                lines.append(f"{indent}{child_id}[{child_type}]")
                lines.append(f"{indent}{parent_id} -->|{slot_name}[{i}]| {child_id}")
                if depth < 3:
                    _add_children(lines, c, child_id, depth + 1)
        else:
            child_id = _safe_id(f"{parent_id}_{slot_name}")
            child_type = getattr(child, "block_type", type(child).__name__)
            indent = "    " * (depth + 1)
            lines.append(f"{indent}{child_id}[{child_type}]")
            lines.append(f"{indent}{parent_id} -->|{slot_name}| {child_id}")
            if depth < 3:
                _add_children(lines, child, child_id, depth + 1)


def _safe_id(s: str) -> str:
    """Convert string to a valid Mermaid node ID."""
    return s.replace("/", "_").replace("-", "_").replace(" ", "_").replace(".", "_")


def config_to_mermaid(config: dict, title: str = "Graph") -> str:
    """Generate Mermaid diagram from a config dict."""
    lines = [f"graph TB"]
    
    def _process(cfg, parent_id=None, depth=0):
        if not isinstance(cfg, dict):
            return
        
        block_type = cfg.get("type", "unknown")
        node_id = _safe_id(f"node_{depth}_{block_type}")
        indent = "    " * (depth + 1)
        
        lines.append(f"{indent}{node_id}[{block_type}]")
        
        if parent_id:
            lines.append(f"{indent}{parent_id} --> {node_id}")
        
        for key, value in cfg.items():
            if key == "type":
                continue
            if isinstance(value, dict) and "type" in value:
                _process(value, node_id, depth + 1)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict) and "type" in item:
                        _process(item, node_id, depth + 1)
    
    _process(config)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="YggDrasil Graph Visualizer")
    parser.add_argument("--config", "-c", help="Path to YAML config")
    parser.add_argument("--output", "-o", help="Output file path", default="graph.md")
    parser.add_argument("--format", "-f", choices=["mermaid", "graphviz"], default="mermaid")
    
    args = parser.parse_args()
    
    if args.config:
        from omegaconf import OmegaConf
        config = OmegaConf.to_container(OmegaConf.load(args.config))
        diagram = config_to_mermaid(config)
    else:
        # Show a sample diagram
        diagram = config_to_mermaid({
            "type": "engine/sampler",
            "model": {
                "type": "model/modular",
                "backbone": {"type": "backbone/unet2d_condition"},
                "codec": {"type": "codec/autoencoder_kl"},
                "conditioner": [{"type": "conditioner/clip_text"}],
                "guidance": [{"type": "guidance/cfg"}],
            },
            "solver": {"type": "diffusion/solver/ddim"},
            "diffusion_process": {"type": "diffusion/process/ddpm"},
        })
    
    output = f"```mermaid\n{diagram}\n```"
    
    with open(args.output, "w") as f:
        f.write(output)
    
    print(f"Graph saved to {args.output}")
    print(diagram)


if __name__ == "__main__":
    main()
