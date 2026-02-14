"""High-level Pipeline API — like HuggingFace Diffusers but with full Lego access.

    # One-liner
    pipe = Pipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", device="cuda")
    image = pipe("a beautiful cat", num_steps=28, seed=42).images[0]

    # From template
    pipe = Pipeline.from_template("sd15_txt2img", device="cuda")
    image = pipe("a cat").images[0]

    # Full graph access underneath
    pipe.graph.nodes["denoise_loop"].graph.replace_node("solver", my_solver)

    # Then generate again
    image = pipe("a cat with new solver").images[0]
"""
from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


@dataclass
class PipelineOutput:
    """Structured output from Pipeline execution.
    
    Attributes:
        images:  List of PIL Images (for image pipelines)
        latents: Raw latent tensors
        audio:   Audio waveform tensors (for audio pipelines)
        video:   Video frame tensors (for video pipelines)
        raw:     Raw graph output dict
    """
    images: Optional[List[Any]] = None
    latents: Optional[torch.Tensor] = None
    audio: Optional[torch.Tensor] = None
    video: Optional[torch.Tensor] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """High-level pipeline wrapping a ComputeGraph.
    
    Provides a diffusers-like interface while maintaining full Lego access
    to the underlying graph.
    """
    
    def __init__(self, graph, *, device=None, dtype=None):
        from yggdrasil.core.graph.graph import ComputeGraph
        self.graph: ComputeGraph = graph
        if device is not None:
            self.graph.to(device, dtype)
    
    @classmethod
    def from_template(cls, template_name: str, *, device=None, dtype=None, **kwargs) -> Pipeline:
        """Create pipeline from a named template.
        
        Args:
            template_name: Template name ("sd15_txt2img", "flux_txt2img", etc.)
            device: Target device ("cuda", "mps", "cpu")
            dtype: Data type (auto-selected if None)
            **kwargs: Extra args for template builder (e.g. pretrained="...")
        
        Returns:
            Pipeline ready for generation.
        
        Example::
        
            pipe = Pipeline.from_template("sd15_txt2img", device="cuda")
            output = pipe("a cat")
        """
        from yggdrasil.core.graph.graph import ComputeGraph
        graph = ComputeGraph.from_template(template_name, **kwargs)
        return cls(graph, device=device, dtype=dtype)
    
    @classmethod
    def from_pretrained(cls, model_id: str, *, device=None, dtype=None, **kwargs) -> Pipeline:
        """Load pipeline from HuggingFace model ID or local path.
        
        Resolution chain (tries in order, collects all errors):
        1. DiffusersBridge.from_pretrained()
        2. Hub registry (resolve_model)
        3. Template string matching
        
        Args:
            model_id: HuggingFace model ID or local path
            device: Target device
            dtype: Data type
        
        Returns:
            Pipeline ready for generation.
        
        Raises:
            ValueError: If model cannot be resolved (with full error chain).
        
        Example::
        
            pipe = Pipeline.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5",
                device="cuda"
            )
        """
        errors = []
        
        # 1. Try DiffusersBridge
        try:
            from yggdrasil.integration.diffusers import DiffusersBridge
            graph = DiffusersBridge.from_pretrained(model_id, **kwargs)
            return cls(graph, device=device, dtype=dtype)
        except Exception as e:
            errors.append(f"DiffusersBridge: {type(e).__name__}: {e}")
        
        # 2. Try hub registry
        try:
            from yggdrasil.hub import resolve_model
            template_name, template_kwargs = resolve_model(model_id)
            return cls.from_template(template_name, device=device, dtype=dtype, **template_kwargs)
        except Exception as e:
            errors.append(f"Hub registry: {type(e).__name__}: {e}")
        
        # 3. Fallback: guess template from model_id
        template_map = {
            "stable-diffusion-v1-5": "sd15_txt2img",
            "stable-diffusion-xl": "sdxl_txt2img",
            "stable-diffusion-3": "sd3_txt2img",
            "flux": "flux_txt2img",
        }
        for key, template in template_map.items():
            if key in model_id.lower():
                try:
                    return cls.from_template(
                        template, device=device, dtype=dtype,
                        pretrained=model_id, **kwargs,
                    )
                except Exception as e:
                    errors.append(f"Template match '{template}': {type(e).__name__}: {e}")
        
        # All methods failed — raise informative error
        error_chain = "\n  ".join(f"{i+1}. {e}" for i, e in enumerate(errors))
        raise ValueError(
            f"Cannot resolve model '{model_id}'. "
            f"Tried {len(errors)} methods:\n  {error_chain}\n"
            f"Tip: Use Pipeline.from_template('template_name') for direct template access, "
            f"or Pipeline.from_graph(graph) for a custom graph."
        )
    
    @classmethod
    def from_graph(cls, graph, *, device=None, dtype=None) -> Pipeline:
        """Create pipeline from an existing ComputeGraph."""
        return cls(graph, device=device, dtype=dtype)
    
    @classmethod
    def from_workflow(cls, path: str, *, device=None, dtype=None, **overrides) -> Pipeline:
        """Load pipeline from a saved workflow file.
        
        Args:
            path: Path to workflow file (.yaml or .json)
            device: Target device
            dtype: Data type
            **overrides: Override saved parameters
        
        Returns:
            Pipeline ready for generation.
        """
        from yggdrasil.core.graph.graph import ComputeGraph
        graph, params = ComputeGraph.from_workflow(path)
        pipe = cls(graph, device=device, dtype=dtype)
        pipe._workflow_params = {**params, **overrides}
        return pipe
    
    @staticmethod
    def list_available() -> Dict[str, Dict[str, Any]]:
        """List all available pipeline templates and supported models.
        
        Returns:
            Dict of {template_name: {description, modality, models, ...}}
        
        Example::
        
            templates = Pipeline.list_available()
            for name, info in templates.items():
                print(f"{name}: {info.get('description', '')}")
        """
        from yggdrasil.core.graph.graph import ComputeGraph
        
        available = {}
        
        # Get all registered templates
        try:
            from yggdrasil.core.graph.templates import image_pipelines
            # Scan the module for builder functions
            for attr_name in dir(image_pipelines):
                if attr_name.startswith("build_"):
                    template_name = attr_name[len("build_"):]
                    func = getattr(image_pipelines, attr_name)
                    doc = (func.__doc__ or "").strip().split("\n")[0]
                    available[template_name] = {
                        "description": doc,
                        "builder": attr_name,
                        "modality": _guess_modality(template_name),
                    }
        except ImportError:
            pass
        
        # Also check the template registry if it exists
        try:
            templates = ComputeGraph._template_registry
            for name, builder in templates.items():
                if name not in available:
                    doc = (builder.__doc__ or "").strip().split("\n")[0]
                    available[name] = {
                        "description": doc,
                        "builder": builder.__name__,
                        "modality": _guess_modality(name),
                    }
        except AttributeError:
            pass
        
        # Add hub models
        try:
            from yggdrasil.hub import MODEL_REGISTRY
            for model_id, info in MODEL_REGISTRY.items():
                template = info[0] if isinstance(info, tuple) else info.get("template", "")
                if template in available:
                    models = available[template].setdefault("models", [])
                    models.append(model_id)
        except (ImportError, AttributeError):
            pass
        
        return available
    
    def __call__(
        self,
        prompt: Union[str, Dict[str, Any], None] = None,
        *,
        negative_prompt: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        num_steps: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> PipelineOutput:
        """Generate output — universal for any modality.
        
        This method introspects the graph to determine what inputs are needed.
        For image pipelines, prompt/width/height/seed apply. For audio, molecular,
        or custom pipelines, use **kwargs to pass graph inputs directly.
        
        Args:
            prompt: Text prompt (str) or structured input (dict).
                    Ignored if the graph has no 'prompt' input.
            negative_prompt: Negative prompt for CFG
            guidance_scale: CFG scale (override graph default)
            num_steps: Denoising steps (override graph default)
            width: Output width in pixels (image pipelines)
            height: Output height in pixels (image pipelines)
            seed: Random seed for reproducibility
            batch_size: Number of outputs to generate
            **kwargs: Additional graph inputs (passed directly to graph)
        
        Returns:
            PipelineOutput with .images, .latents, .audio, .video, .raw
        
        Examples::
        
            # Image generation
            output = pipe("a beautiful cat", guidance_scale=7.5, seed=42)
            image = output.images[0]
            
            # Audio generation (custom pipeline)
            output = pipe(spectrogram=my_input, seed=42)
            
            # Arbitrary inputs via kwargs
            output = pipe(molecule=smiles_str, temperature=0.8)
        """
        meta = self.graph.metadata
        graph_input_names = set(self.graph.graph_inputs.keys())
        
        # ── Apply runtime overrides to graph nodes ──
        if guidance_scale is not None:
            self._apply_guidance_scale(guidance_scale)
        if num_steps is not None:
            self._apply_num_steps(num_steps)
        
        # ── Prepare prompt (only if graph expects it) ──
        if prompt is not None:
            if isinstance(prompt, str):
                prompt = {"text": prompt}
            kwargs["prompt"] = prompt
        
        # ── Generate noise (only if graph expects latents and none provided) ──
        if "latents" in graph_input_names and "latents" not in kwargs:
            if width is None:
                width = meta.get("default_width", 512)
            if height is None:
                height = meta.get("default_height", 512)
            kwargs["latents"] = self._make_noise(
                batch_size=batch_size, width=width, height=height, seed=seed,
            )
        
        # ── Execute graph (non-strict: tolerate missing optional inputs) ──
        from yggdrasil.core.graph.executor import GraphExecutor
        raw = GraphExecutor(strict=False).execute(self.graph, **kwargs)
        
        # ── Build output ──
        return self._build_output(raw)
    
    def to(self, device=None, dtype=None) -> Pipeline:
        """Move pipeline to device."""
        self.graph.to(device, dtype)
        return self
    
    # ── Internal helpers ──
    
    def _apply_guidance_scale(self, scale: float):
        for _, block in self.graph._iter_all_blocks():
            bt = getattr(block, 'block_type', '')
            if 'guidance' in bt and hasattr(block, 'scale'):
                block.scale = scale
    
    def _apply_num_steps(self, num_steps: int):
        for _, block in self.graph._iter_all_blocks():
            if hasattr(block, 'num_iterations'):
                block.num_iterations = num_steps
    
    def _make_noise(self, batch_size=1, width=512, height=512, seed=None):
        meta = self.graph.metadata
        channels = meta.get("latent_channels", 4)
        scale = meta.get("spatial_scale_factor", 8)
        h, w = height // scale, width // scale
        
        device = self.graph._device or torch.device("cpu")
        device_type = device.type if hasattr(device, 'type') else str(device)
        
        if seed is not None:
            if device_type == "mps":
                g = torch.Generator().manual_seed(seed)
                noise = torch.randn(batch_size, channels, h, w, generator=g)
            else:
                g = torch.Generator(device).manual_seed(seed)
                noise = torch.randn(batch_size, channels, h, w, device=device, generator=g)
        else:
            noise = torch.randn(batch_size, channels, h, w)
        
        return noise.to(device)
    
    def _build_output(self, raw: Dict[str, Any]) -> PipelineOutput:
        """Convert raw graph output to PipelineOutput."""
        output = PipelineOutput(raw=raw)
        
        # Extract latents
        if "latents" in raw:
            output.latents = raw["latents"]
        
        # Try to convert decoded tensor to images
        decoded = raw.get("decoded")
        if decoded is not None and isinstance(decoded, torch.Tensor):
            if decoded.dim() == 4 and decoded.shape[1] in (1, 3, 4):
                output.images = self._tensor_to_images(decoded)
            elif decoded.dim() == 5:
                output.video = decoded
            elif decoded.dim() <= 2:
                output.audio = decoded
        
        # If no decoded, check for raw output
        if output.images is None and output.audio is None and output.video is None:
            out_tensor = raw.get("output")
            if isinstance(out_tensor, torch.Tensor):
                if out_tensor.dim() == 4:
                    output.images = self._tensor_to_images(out_tensor)
        
        return output
    
    @staticmethod
    def _tensor_to_images(tensor: torch.Tensor):
        """Convert tensor [-1,1] or [0,1] to list of PIL Images."""
        try:
            from PIL import Image
        except ImportError:
            return None
        
        img = tensor.cpu().float()
        if img.min() < -0.5:
            img = (img / 2 + 0.5)
        img = img.clamp(0, 1)
        
        images = []
        for i in range(img.shape[0]):
            arr = (img[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            if arr.shape[-1] == 1:
                arr = arr.squeeze(-1)
            images.append(Image.fromarray(arr))
        
        return images
    
    def __repr__(self):
        device = self.graph._device or "cpu"
        return f"<Pipeline '{self.graph.name}' device={device} nodes={len(self.graph.nodes)}>"


def _guess_modality(template_name: str) -> str:
    """Guess modality from template name."""
    name = template_name.lower()
    if any(k in name for k in ["img2img", "txt2img", "inpaint", "upscale"]):
        return "image"
    if any(k in name for k in ["video", "wan", "animate"]):
        return "video"
    if any(k in name for k in ["audio", "music", "speech"]):
        return "audio"
    if any(k in name for k in ["3d", "mesh", "point_cloud"]):
        return "3d"
    return "image"  # default
