"""Config-driven execution — run any workflow from a file.

Usage::

    # From Python
    from yggdrasil.runner import Runner
    output = Runner.execute("workflow.yaml")
    output = Runner.execute("workflow.json", seed=42)  # override params
    
    # From command line
    python -m yggdrasil.runner workflow.yaml --seed 42

This is YggDrasil's equivalent of ComfyUI's JSON workflow replay:
save a complete pipeline + parameters, then reproduce exact results.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Runner:
    """Execute workflows from config files (YAML/JSON).
    
    A workflow file contains:
    - Graph structure (nodes, edges, inputs, outputs)
    - Runtime parameters (prompt, seed, guidance_scale, etc.)
    - Metadata (model versions, creation date)
    
    Example workflow.yaml::
    
        name: my_generation
        nodes:
          clip:
            block: conditioner/clip_text
            config: {pretrained: openai/clip-vit-large-patch14}
          backbone:
            block: backbone/unet2d_condition
            config: {}
        edges:
          - [clip.embedding, backbone.condition]
        inputs:
          prompt: [clip, text]
          latents: [backbone, x]
        outputs:
          result: [backbone, output]
        parameters:
          prompt: {text: "a beautiful cat"}
          guidance_scale: 7.5
          seed: 42
    """
    
    @staticmethod
    def execute(config_path: str | Path, **overrides) -> Any:
        """Execute a workflow from file.
        
        Args:
            config_path: Path to workflow file (.yaml or .json)
            **overrides: Override saved parameters (e.g. seed=42)
        
        Returns:
            PipelineOutput with generation results.
        
        Example::
        
            # Basic
            output = Runner.execute("workflow.yaml")
            
            # Override seed
            output = Runner.execute("workflow.yaml", seed=123)
            
            # Override prompt
            output = Runner.execute("workflow.yaml", prompt={"text": "new prompt"})
        """
        from yggdrasil.core.graph.graph import ComputeGraph
        from yggdrasil.pipeline import InferencePipeline
        
        graph, parameters = ComputeGraph.from_workflow(config_path)
        
        # Merge overrides into parameters
        parameters.update(overrides)
        
        logger.info(f"Running workflow '{graph.name}' with {len(graph.nodes)} nodes")
        logger.info(f"Parameters: {list(parameters.keys())}")
        
        pipe = InferencePipeline.from_graph(graph)
        return pipe(**parameters)
    
    @staticmethod
    def execute_raw(config_path: str | Path, **overrides) -> Dict[str, Any]:
        """Execute a workflow and return raw graph outputs (no InferencePipeline wrapping).

        Useful for non-standard pipelines that don't fit the InferencePipeline model.
        
        Args:
            config_path: Path to workflow file
            **overrides: Override saved parameters
            
        Returns:
            Dict of raw graph outputs.
        """
        from yggdrasil.core.graph.graph import ComputeGraph
        from yggdrasil.core.graph.executor import GraphExecutor
        
        graph, parameters = ComputeGraph.from_workflow(config_path)
        parameters.update(overrides)
        
        executor = GraphExecutor(strict=False)
        return executor.execute(graph, **parameters)
    
    @staticmethod
    def validate(config_path: str | Path) -> Dict[str, Any]:
        """Validate a workflow file without executing it.
        
        Returns:
            Dict with 'valid' (bool), 'errors' (list), 'info' (dict).
        """
        from yggdrasil.core.graph.graph import ComputeGraph
        
        try:
            graph, parameters = ComputeGraph.from_workflow(config_path)
            errors = graph.validate()
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "info": {
                    "name": graph.name,
                    "nodes": len(graph.nodes),
                    "edges": len(graph.edges),
                    "inputs": list(graph.graph_inputs.keys()),
                    "outputs": list(graph.graph_outputs.keys()),
                    "parameters": list(parameters.keys()),
                },
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"{type(e).__name__}: {e}"],
                "info": {},
            }


# CLI entry point
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="YggDrasil workflow runner",
        usage="python -m yggdrasil.runner <workflow.yaml> [--param value ...]",
    )
    parser.add_argument("workflow", help="Path to workflow file (YAML or JSON)")
    parser.add_argument("--validate", action="store_true", help="Only validate, don't execute")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    
    args, unknown = parser.parse_known_args()
    
    # Parse extra --key value pairs as overrides
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                overrides[key] = unknown[i + 1]
                i += 2
            else:
                overrides[key] = True
                i += 1
        else:
            i += 1
    
    if args.validate:
        result = Runner.validate(args.workflow)
        if result["valid"]:
            print(f"Workflow is valid: {result['info']}")
        else:
            print(f"Workflow has errors: {result['errors']}")
        return
    
    logging.basicConfig(level=logging.INFO)
    output = Runner.execute(args.workflow, **overrides)
    
    # Save results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if hasattr(output, 'images') and output.images:
        for i, img in enumerate(output.images):
            img.save(out_dir / f"output_{i}.png")
            print(f"Saved: {out_dir / f'output_{i}.png'}")
    
    print(f"Done. Raw outputs: {list(output.raw.keys())}")


if __name__ == "__main__":
    main()
