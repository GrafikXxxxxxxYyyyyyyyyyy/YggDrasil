"""RunPod serverless handler for YggDrasil.

Supports both legacy ModularDiffusionModel and new ComputeGraph.
"""
from __future__ import annotations

import torch
import json
from typing import Dict, Any


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod serverless handler.
    
    Receives generation requests and returns results.
    Deploy with: runpodctl deploy --handler yggdrasil.deployment.cloud.runpod
    
    Expected input for graph-based execution::
    
        {
            "input": {
                "mode": "graph",  # "graph" or "legacy"
                "template": "sd15_txt2img",  # or "graph_yaml" for custom
                "prompt": "a beautiful landscape",
                "num_steps": 50,
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512,
                "seed": -1
            }
        }
    """
    try:
        input_data = event.get("input", {})
        mode = input_data.get("mode", "auto")
        
        if mode == "graph" or "template" in input_data:
            return _handle_graph(input_data)
        else:
            return _handle_legacy(input_data)
    
    except Exception as e:
        return {"error": str(e), "status": "error"}


def _handle_graph(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle graph-based generation."""
    from yggdrasil.core.graph.graph import ComputeGraph
    from yggdrasil.core.graph.executor import GraphExecutor
    
    # Load or build graph
    if not hasattr(_handle_graph, "_cache"):
        _handle_graph._cache = {}
    
    template = input_data.get("template", "sd15_txt2img")
    graph_yaml = input_data.get("graph_yaml")
    
    cache_key = template or "custom"
    
    if cache_key not in _handle_graph._cache:
        if graph_yaml:
            import tempfile, os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(graph_yaml)
                graph = ComputeGraph.from_yaml(f.name)
                os.unlink(f.name)
        else:
            graph = ComputeGraph.from_template(template)
        
        # Move to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for name, block in graph.nodes.items():
            if hasattr(block, 'to'):
                block.to(device)
        
        _handle_graph._cache[cache_key] = graph
    
    graph = _handle_graph._cache[cache_key]
    
    # Prepare inputs
    prompt = input_data.get("prompt", "")
    width = input_data.get("width", 512)
    height = input_data.get("height", 512)
    seed = input_data.get("seed", -1)
    
    graph_inputs = {
        "prompt": {"text": prompt},
        "latents": _generate_noise(1, 4, height // 8, width // 8, seed),
        "timestep": torch.tensor([999]),
    }
    
    # Execute graph
    executor = GraphExecutor()
    outputs = executor.execute(graph, **graph_inputs)
    
    # Convert output to image
    result = outputs.get("decoded", outputs.get("image", outputs.get("next_latents")))
    if result is not None:
        img_b64 = _tensor_to_base64(result)
        return {
            "output": {"image_base64": img_b64, "width": width, "height": height},
            "status": "success",
        }
    
    return {"output": outputs, "status": "success"}


def _handle_legacy(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle legacy model-based generation."""
    from yggdrasil.assemblers import ModelAssembler, PipelineAssembler
    
    model_id = input_data.get("model_id", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    
    if not hasattr(_handle_legacy, "_model_cache"):
        _handle_legacy._model_cache = {}
    
    if model_id not in _handle_legacy._model_cache:
        model = ModelAssembler.from_pretrained(model_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        _handle_legacy._model_cache[model_id] = model
    
    model = _handle_legacy._model_cache[model_id]
    
    sampler = PipelineAssembler.for_generation(
        model=model,
        num_steps=input_data.get("num_steps", 50),
        guidance_scale=input_data.get("guidance_scale", 7.5),
    )
    
    condition = {"text": input_data.get("prompt", "")}
    width = input_data.get("width", 512)
    height = input_data.get("height", 512)
    
    codec = model._slot_children.get("codec")
    if codec and hasattr(codec, "get_latent_shape"):
        shape = codec.get_latent_shape(1, height, width)
    else:
        shape = (1, 4, height // 8, width // 8)
    
    seed = input_data.get("seed", -1)
    generator = torch.Generator().manual_seed(seed) if seed >= 0 else None
    
    result = sampler.sample(condition=condition, shape=shape, generator=generator)
    img_b64 = _tensor_to_base64(result)
    
    return {
        "output": {"image_base64": img_b64, "width": width, "height": height},
        "status": "success",
    }


def _generate_noise(batch, channels, h, w, seed=-1):
    """Generate initial noise tensor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed >= 0:
        generator = torch.Generator(device).manual_seed(seed)
        return torch.randn(batch, channels, h, w, device=device, generator=generator)
    return torch.randn(batch, channels, h, w, device=device)


def _tensor_to_base64(tensor):
    """Convert output tensor to base64 PNG."""
    import base64
    from io import BytesIO
    from PIL import Image
    import numpy as np
    
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    image = tensor.cpu().float()
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).clip(0, 255).astype(np.uint8)
    
    pil_image = Image.fromarray(image)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# RunPod serverless entry point
if __name__ == "__main__":
    try:
        import runpod
        runpod.serverless.start({"handler": handler})
    except ImportError:
        print("runpod package not installed. Install with: pip install runpod")
