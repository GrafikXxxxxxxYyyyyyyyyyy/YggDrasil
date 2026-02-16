"""High-level InferencePipeline API — like HuggingFace Diffusers but with full Lego access.

    # One-liner
    pipe = InferencePipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", device="cuda")
    image = pipe("a beautiful cat", num_steps=28, seed=42).images[0]

    # From template
    pipe = InferencePipeline.from_template("sd15_txt2img", device="cuda")
    image = pipe("a cat").images[0]

    # Full graph access underneath
    pipe.graph.nodes["denoise_loop"].graph.replace_node("solver", my_solver)

    # Then generate again
    image = pipe("a cat with new solver").images[0]
"""
from __future__ import annotations

import io
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError


def load_image_from_url_or_path(source: Union[str, Path]) -> Optional[Any]:
    """Загрузить изображение по URL или с диска. Возвращает PIL.Image или None при ошибке.

    source: http(s)://... или путь к файлу.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    source = str(source).strip()
    if not source:
        return None
    try:
        if source.startswith(("http://", "https://")):
            with urlopen(source, timeout=30) as resp:
                data = resp.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
        else:
            img = Image.open(source).convert("RGB")
        return img
    except (URLError, OSError, Exception):
        return None


def _pil_to_tensor(pil_img: Any) -> torch.Tensor:
    """PIL Image -> tensor (1, 3, H, W) float32 [0, 1]."""
    arr = np.array(pil_img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def _normalize_ip_image(
    value: Any,
) -> List[Any]:
    """Normalize IP-Adapter input to a list of images (PIL or tensor).

    Accepts:
        - Single: Path, URL (str), PIL Image, tensor (3,H,W) or (1,3,H,W), or dict with "image"/"images".
        - List/tuple: list of any of the above (paths, URLs, PILs, tensors).
        - Dict (vocabulary): mapping of keys to images (e.g. {"style": url1, "face": url2}); values as above.

    Returns:
        List of PIL Images or tensors (already on correct device if tensor). Empty list if nothing valid.
    """
    try:
        from PIL import Image
    except ImportError:
        Image = None

    def is_pil(x):
        return Image is not None and hasattr(x, "size") and hasattr(x, "mode")

    def is_tensor_img(x):
        if not isinstance(x, torch.Tensor):
            return False
        return x.dim() in (3, 4) and x.shape[-3] == 3

    def one_to_item(item: Any) -> Optional[Any]:
        if item is None:
            return None
        if is_pil(item) or is_tensor_img(item):
            return item
        if isinstance(item, (str, Path)):
            out = load_image_from_url_or_path(item)
            return out
        if isinstance(item, dict):
            img = item.get("image") or (item.get("images") and item["images"][0])
            return one_to_item(img) if img is not None else None
        return None

    if value is None:
        return []
    if isinstance(value, dict) and ("image" in value or "images" in value):
        img = value.get("image")
        imgs = value.get("images", [])
        if img is not None and not isinstance(img, (list, tuple)):
            v = one_to_item(img)
            return [v] if v is not None else []
        if isinstance(img, (list, tuple)):
            imgs = img
        if isinstance(imgs, (list, tuple)):
            out = []
            for i in imgs:
                v = one_to_item(i)
                if v is not None:
                    out.append(v)
            return out
        return []
    if isinstance(value, (list, tuple)):
        out = []
        for i in value:
            v = one_to_item(i)
            if v is not None:
                out.append(v)
        return out
    if isinstance(value, dict):
        # Vocabulary: key -> image
        out = []
        for v in value.values():
            x = one_to_item(v)
            if x is not None:
                out.append(x)
        return out
    v = one_to_item(value)
    return [v] if v is not None else []


@dataclass
class PipelineOutput:
    """Structured output from InferencePipeline execution.
    
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


class InferencePipeline:
    """High-level inference pipeline wrapping a ComputeGraph.

    Provides a diffusers-like interface while maintaining full Lego access
    to the underlying graph. Accepts either a single graph or a list/dict of
    graphs (combined pipeline); ref REFACTORING_GRAPH_PIPELINE_ENGINE.md §9.
    """

    def __init__(
        self,
        graph=None,
        *,
        graphs=None,
        device=None,
        dtype=None,
        connections=None,
        inputs=None,
        outputs=None,
        name="combined",
        parallel_groups=None,
        merge_strategy=None,
    ):
        """Build pipeline from one graph or from list/dict of graphs (combined).

        Args:
            graph: Single ComputeGraph (legacy: positional or keyword).
            graphs: List of (stage_name, graph_or_template) or dict name->graph/template.
                    If given, builds combined pipeline; 'graph' is ignored.
            device, dtype: Target device/dtype.
            connections: For graphs= dict/list — (src, src_port, dst, dst_port); default linear chain.
            inputs, outputs: Graph-level input/output mapping when using graphs=.
            name: Pipeline graph name when using graphs=.
            parallel_groups: For graphs= — optional list of stage name groups (execution plan; §11.7 P3).
            merge_strategy: For graphs= — optional merge strategy for parallel branches (reserved).
        """
        from yggdrasil.core.graph.graph import ComputeGraph
        from yggdrasil.core.graph.stage import AbstractStage

        if graphs is not None:
            # Combined pipeline: same logic as from_combined
            if isinstance(graphs, dict):
                stages = [(k, v) for k, v in graphs.items()]
            else:
                stages = list(graphs)
            g = ComputeGraph(name)
            for stage_def in stages:
                stage_name = stage_def[0]
                if len(stage_def) == 2:
                    _, graph_or_tpl = stage_def
                    if isinstance(graph_or_tpl, str):
                        inner = ComputeGraph.from_template(graph_or_tpl)
                    else:
                        inner = graph_or_tpl
                else:
                    inner = ComputeGraph.from_template(
                        stage_def[1], **(stage_def[2] if len(stage_def) > 2 else {})
                    )
                stage_block = AbstractStage(config={"type": "stage/abstract"}, graph=inner)
                g.add_node(stage_name, stage_block)
            if connections:
                for edge in connections:
                    if len(edge) >= 4:
                        g.connect(edge[0], edge[1], edge[2], edge[3])
            else:
                names = list(g.nodes.keys())
                for i in range(len(names) - 1):
                    g.connect(names[i], "output", names[i + 1], "input")
            for input_name, mapping in (inputs or {}).items():
                if isinstance(mapping, (list, tuple)) and len(mapping) >= 2:
                    g.expose_input(input_name, str(mapping[0]), str(mapping[1]))
                elif isinstance(mapping, str) and "." in mapping:
                    node, port = mapping.split(".", 1)
                    g.expose_input(input_name, node.strip(), port.strip())
            for output_name, mapping in (outputs or {}).items():
                if isinstance(mapping, (list, tuple)) and len(mapping) >= 2:
                    g.expose_output(output_name, str(mapping[0]), str(mapping[1]))
                elif isinstance(mapping, str) and "." in mapping:
                    node, port = mapping.split(".", 1)
                    g.expose_output(output_name, node.strip(), port.strip())
            if parallel_groups is not None:
                g.metadata["parallel_groups"] = parallel_groups
            if merge_strategy is not None:
                g.metadata["merge_strategy"] = merge_strategy
            self.graph = g
        elif graph is not None:
            self.graph = graph
        else:
            raise ValueError("InferencePipeline requires graph= or graphs=")
        if device is not None:
            self.graph.to(device, dtype)

    @classmethod
    def from_template(cls, template_name: str, *, device=None, dtype=None, **kwargs) -> "InferencePipeline":
        """Create pipeline from a named template.
        
        Args:
            template_name: Template name ("sd15_txt2img", "flux_txt2img", etc.)
            device: Target device ("cuda", "mps", "cpu")
            dtype: Data type (auto-selected if None)
            **kwargs: Extra args for template builder (e.g. pretrained="...")
        
        Returns:
            InferencePipeline ready for generation.

        Example::

            pipe = InferencePipeline.from_template("sd15_txt2img", device="cuda")
            output = pipe("a cat")
        """
        from yggdrasil.core.graph.graph import ComputeGraph
        graph = ComputeGraph.from_template(template_name, **kwargs)
        # Video diffusion models: default to CUDA when available if device not set
        if device is None and graph.metadata.get("modality") == "video" and torch.cuda.is_available():
            device = "cuda"
        return cls(graph, device=device, dtype=dtype)
    
    @classmethod
    def from_pretrained(cls, model_id: str, *, device=None, dtype=None, **kwargs) -> "InferencePipeline":
        """Load pipeline from HuggingFace model ID or local path.
        
        Prefer native YggDrasil graph (from template). Resolution order:
        1. Hub registry (resolve_model) → ComputeGraph.from_template
        2. Template matching from model_id (sd15/sdxl/sd3/flux, stable-diffusion-*)
        3. DiffusersBridge.from_pretrained (load diffusers then import to graph) — last resort
        
        Args:
            model_id: HuggingFace model ID or local path
            device: Target device
            dtype: Data type
        
        Returns:
            InferencePipeline with a YggDrasil ComputeGraph.

        Raises:
            ValueError: If model cannot be resolved (with full error chain).
        
        Example::
        
            pipe = InferencePipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                device="cuda"
            )
        """
        errors = []
        
        # 1. Prefer native YggDrasil graph: hub registry
        try:
            from yggdrasil.hub import resolve_model
            template_name, template_kwargs = resolve_model(model_id)
            return cls.from_template(template_name, device=device, dtype=dtype, **template_kwargs)
        except Exception as e:
            errors.append(f"Hub registry: {type(e).__name__}: {e}")
        
        # 2. Prefer native YggDrasil graph: template from model_id (preset or keyword)
        preset_to_template = {"sd15": "sd15_txt2img", "sdxl": "sdxl_txt2img", "sd3": "sd3_txt2img", "flux": "flux_txt2img"}
        if model_id.lower() in preset_to_template:
            try:
                return cls.from_template(
                    preset_to_template[model_id.lower()],
                    device=device,
                    dtype=dtype,
                    **kwargs,
                )
            except Exception as e:
                errors.append(f"Preset '{model_id}': {type(e).__name__}: {e}")
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
        
        # 3. Fallback: load via diffusers and convert to YggDrasil graph
        try:
            from yggdrasil.integration.diffusers import DiffusersBridge
            graph = DiffusersBridge.from_pretrained(model_id, **kwargs)
            return cls(graph, device=device, dtype=dtype)
        except Exception as e:
            errors.append(f"DiffusersBridge: {type(e).__name__}: {e}")
        
        # All methods failed — raise informative error
        error_chain = "\n  ".join(f"{i+1}. {e}" for i, e in enumerate(errors))
        raise ValueError(
            f"Cannot resolve model '{model_id}'. "
            f"Tried {len(errors)} methods:\n  {error_chain}\n"
            f"Tip: Use InferencePipeline.from_template('template_name') for direct template access, "
            f"or InferencePipeline.from_graph(graph) for a custom graph."
        )
    
    @classmethod
    def from_graph(cls, graph, *, device=None, dtype=None) -> "InferencePipeline":
        """Create pipeline from an existing ComputeGraph."""
        return cls(graph, device=device, dtype=dtype)

    @classmethod
    def from_spec(
        cls,
        spec: Union[Any, List[Any], Dict[str, Any], str, Path],
        *,
        device=None,
        dtype=None,
        **kwargs,
    ) -> "InferencePipeline":
        """Unified entry point: create pipeline from graph, list of graphs, dict of graphs, or config path.

        Dispatches by type of spec:
        - ComputeGraph -> from_graph(spec)
        - list (of stages) -> from_combined(stages=spec)
        - dict (name -> graph or template) -> from_combined(stages=[(k,v) for k,v in spec.items()], ...)
        - str | Path -> from_config(spec)

        Args:
            spec: Graph, list of (name, graph/template), dict of name->graph/template, or path to YAML/JSON.
            device, dtype: Target device/dtype.
            **kwargs: Passed to from_combined when spec is list/dict (e.g. links, inputs, outputs, name).

        Returns:
            InferencePipeline

        Example::
            pipe = InferencePipeline.from_spec("config.yaml", device="cuda")
            pipe = InferencePipeline.from_spec(my_graph, device="cuda")
            pipe = InferencePipeline.from_spec([("stage0", "sd15_txt2img"), ("stage1", "codec/vae")], device="cuda")
            pipe = InferencePipeline.from_spec({"gen": g1, "refine": g2}, links=[...], device="cuda")
        """
        from yggdrasil.core.graph.graph import ComputeGraph

        if isinstance(spec, ComputeGraph):
            return cls.from_graph(spec, device=device, dtype=dtype)
        if isinstance(spec, (list, tuple)):
            return cls.from_combined(stages=spec, device=device, dtype=dtype, **kwargs)
        if isinstance(spec, dict):
            stages = [(k, v) for k, v in spec.items()]
            return cls.from_combined(stages=stages, device=device, dtype=dtype, **kwargs)
        if isinstance(spec, (str, Path)):
            return cls.from_config(spec, device=device, dtype=dtype, **kwargs)
        raise TypeError(
            f"InferencePipeline.from_spec(spec) expects ComputeGraph, list, dict, or path; got {type(spec).__name__}"
        )

    @classmethod
    def from_diffusers(cls, pipe: Any, *, device=None, dtype=None) -> "InferencePipeline":
        """Wrap an already-loaded HuggingFace Diffusers pipeline as InferencePipeline.
        
        Uses DiffusersBridge.import_pipeline to build a ComputeGraph (single Solver,
        no separate Scheduler). The resulting graph is \"flat\" (one step per execute);
        InferencePipeline runs the denoising loop automatically.
        
        Example::
        
            from diffusers import StableDiffusionPipeline
            pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
            ygg = InferencePipeline.from_diffusers(pipe, device="cuda")
            image = ygg("a cat", num_steps=28).images[0]
        """
        from yggdrasil.integration.diffusers import DiffusersBridge
        graph = DiffusersBridge.import_pipeline(pipe)
        return cls(graph, device=device, dtype=dtype)

    @classmethod
    def from_config(cls, path: Union[str, Path], *, device=None, dtype=None, **overrides) -> "InferencePipeline":
        """Load pipeline from YAML/JSON config.
        
        Supports:
        - Single graph: nodes/edges/inputs/outputs.
        - Combined pipeline: kind: combined_pipeline, stages + links (each node is an AbstractStage).
        
        Args:
            path: Path to .yaml or .json config file
            device: Target device
            dtype: Data type
            **overrides: Not used; reserved for future overrides
        
        Returns:
            InferencePipeline
        """
        from yggdrasil.core.graph.graph import ComputeGraph
        path = Path(path)
        graph = ComputeGraph.from_yaml(path)
        return cls(graph, device=device, dtype=dtype)

    @classmethod
    def from_combined(
        cls,
        stages: List[Union[Tuple[str, Any], Tuple[str, str], Tuple[str, str, Dict]]],
        links: Optional[List[Tuple[str, str, str, str]]] = None,
        inputs: Optional[Dict[str, Union[str, List]]] = None,
        outputs: Optional[Dict[str, Union[str, List]]] = None,
        *,
        name: str = "combined",
        parallel_groups: Optional[List[List[str]]] = None,
        merge_strategy: Optional[Union[Dict[str, Any], str]] = None,
        device=None,
        dtype=None,
    ) -> "InferencePipeline":
        """Build pipeline from a list of stages (graphs or templates) and optional links.
        
        Args:
            stages: List of (stage_name, graph_or_template) or (stage_name, template_name, kwargs).
                   Each stage is wrapped in AbstractStage.
            links: List of (src_node, src_port, dst_node, dst_port). Default: chain stage0.output -> stage1.input.
            inputs: Graph inputs, e.g. {"prompt": ["stage0", "text"]} or {"prompt": "stage0.prompt"}.
            outputs: Graph outputs, e.g. {"images": ["stage_last", "output"]}.
            name: Pipeline graph name.
            parallel_groups: Optional list of groups; each group is a list of stage names that may run in parallel
                             (execution plan: level-by-level). §11.7 P3.
            merge_strategy: Optional dict or strategy id for merging outputs of parallel branches (reserved).
            device, dtype: Target device/dtype.
        
        Returns:
            InferencePipeline
        """
        from yggdrasil.core.graph.graph import ComputeGraph
        from yggdrasil.core.graph.stage import AbstractStage

        graph = ComputeGraph(name)
        for i, stage_def in enumerate(stages):
            stage_name = stage_def[0]
            if len(stage_def) == 2:
                _, graph_or_tpl = stage_def
                if isinstance(graph_or_tpl, str):
                    inner = ComputeGraph.from_template(graph_or_tpl)
                else:
                    inner = graph_or_tpl  # already ComputeGraph
            else:
                inner = ComputeGraph.from_template(stage_def[1], **(stage_def[2] if len(stage_def) > 2 else {}))
            stage_block = AbstractStage(config={"type": "stage/abstract"}, graph=inner)
            graph.add_node(stage_name, stage_block)
        if links:
            for edge in links:
                if len(edge) >= 4:
                    graph.connect(edge[0], edge[1], edge[2], edge[3])
        else:
            names = list(graph.nodes.keys())
            for i in range(len(names) - 1):
                graph.connect(names[i], "output", names[i + 1], "input")
        inputs = inputs or {}
        outputs = outputs or {}
        for input_name, mapping in inputs.items():
            if isinstance(mapping, (list, tuple)) and len(mapping) >= 2:
                graph.expose_input(input_name, str(mapping[0]), str(mapping[1]))
            elif isinstance(mapping, str) and "." in mapping:
                node, port = mapping.split(".", 1)
                graph.expose_input(input_name, node.strip(), port.strip())
        for output_name, mapping in outputs.items():
            if isinstance(mapping, (list, tuple)) and len(mapping) >= 2:
                graph.expose_output(output_name, str(mapping[0]), str(mapping[1]))
            elif isinstance(mapping, str) and "." in mapping:
                node, port = mapping.split(".", 1)
                graph.expose_output(output_name, node.strip(), port.strip())
        if parallel_groups is not None:
            graph.metadata["parallel_groups"] = parallel_groups
        if merge_strategy is not None:
            graph.metadata["merge_strategy"] = merge_strategy
        return cls(graph, device=device, dtype=dtype)

    @classmethod
    def from_workflow(cls, path: str, *, device=None, dtype=None, **overrides) -> "InferencePipeline":
        """Load pipeline from a saved workflow file.
        
        Args:
            path: Path to workflow file (.yaml or .json)
            device: Target device
            dtype: Data type
            **overrides: Override saved parameters
        
        Returns:
            InferencePipeline ready for generation.
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
        
            templates = InferencePipeline.list_available()
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
        
        # Template registry (all @register_template: image, audio, video, ...)
        try:
            from yggdrasil.core.graph.templates import list_templates
            for name in list_templates():
                if name not in available:
                    from yggdrasil.core.graph.templates import get_template
                    builder = get_template(name)
                    doc = (getattr(builder, "__doc__") or "").strip().split("\n")[0]
                    available[name] = {
                        "description": doc,
                        "builder": getattr(builder, "__name__", name),
                        "modality": _guess_modality(name),
                    }
        except Exception:
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
    
    @staticmethod
    def list_audio_templates() -> Dict[str, str]:
        """Список аудио-шаблонов (имя → краткое описание). Единая система генерации звука."""
        try:
            from yggdrasil.core.graph.templates.audio_pipelines import list_audio_templates
            return list_audio_templates()
        except ImportError:
            return {
                k: v.get("description", k)
                for k, v in InferencePipeline.list_available().items()
                if v.get("modality") == "audio"
            }

    @staticmethod
    def list_video_templates() -> Dict[str, str]:
        """Список видео-шаблонов (имя → краткое описание). Генерация видео и анимация изображений."""
        try:
            from yggdrasil.core.graph.templates.video_pipelines import list_video_templates
            return list_video_templates()
        except ImportError:
            return {
                k: v.get("description", k)
                for k, v in InferencePipeline.list_available().items()
                if v.get("modality") == "video"
            }

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
            control_image: For ControlNet: single image; or list of images (order must match graph: first = first ControlNet, second = second, …); or dict for any order — key by control_type (e.g. {"canny": url1, "depth": url2}) or by input name (e.g. {"control_image": url1, "control_image_controlnet_1": url2}).
            controlnet_scale: Сила ControlNet: число (для всех), список по порядку или словарь. Ключи словаря: control_type (\"depth\", \"canny\"), имя узла (\"controlnet-depth\", \"controlnet-canny\") или \"controlnet-<type>\". По умолчанию 1.0.
            ip_image: IP-Adapter input: single (Path, URL, PIL, or tensor), list of such, or dict vocabulary (key -> image). All images are encoded and combined (e.g. mean-pooled) for one conditioning. Pass when graph has IP-Adapter.
            ip_adapter_scale: Strength of IP-Adapter effect (0.0–1.5+). Higher = output follows reference image more.
                              Default 0.6; 0.8–1.0 for noticeable style; 1.2–1.5 for strong resemblance (e.g. same object/shape).
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
        
        # ── Apply runtime overrides to graph nodes (only when graph has denoise loop; §11.5 N3) ──
        has_denoise_loop = self._get_denoise_loop_node()[0] is not None
        if has_denoise_loop:
            if guidance_scale is not None:
                self._apply_guidance_scale(guidance_scale)
            if num_steps is not None:
                self._apply_num_steps(num_steps)
                kwargs["num_steps"] = num_steps  # so flat diffusers loop can use it
        
        # ── Prepare prompt (only if graph expects it) ──
        if "prompt" in graph_input_names:
            if prompt is None:
                prompt = ""
            if isinstance(prompt, str):
                prompt = {"text": prompt}
            kwargs["prompt"] = prompt
        if "negative_prompt" in graph_input_names:
            kwargs.setdefault("negative_prompt", "")
            if isinstance(kwargs["negative_prompt"], dict):
                kwargs["negative_prompt"] = kwargs["negative_prompt"].get("text", "")
        
        # ── Default width/height from metadata (e.g. SDXL 1024) ──
        if width is None:
            width = meta.get("default_width", 512)
        if height is None:
            height = meta.get("default_height", 512)
        if "height" in graph_input_names and "height" not in kwargs:
            kwargs["height"] = height
        if "width" in graph_input_names and "width" not in kwargs:
            kwargs["width"] = width

        # ── Multiple ControlNets: control_image can be list (by order) or dict (by control_type or input name) ──
        control_inputs, ctype_to_input = self._get_controlnet_input_names_and_types()
        # Fallback: if ctype_to_input empty (e.g. loop not ready at first call), build from blocks order
        if not ctype_to_input and control_inputs:
            for _name, blk in self._get_controlnet_blocks_ordered():
                ct = getattr(blk, "control_type", None)
                if ct is not None:
                    name = next((c for c in control_inputs if c == f"control_image_{_name}" or (c == "control_image" and _name == "controlnet")), None)
                    if name:
                        ctype_to_input[ct] = name
        if "control_image" in kwargs:
            val = kwargs.pop("control_image")
            if val is not None:
                if isinstance(val, (list, tuple)):
                    for i, name in enumerate(control_inputs):
                        if i < len(val):
                            kwargs[name] = val[i]
                elif isinstance(val, dict):
                    for k, v in val.items():
                        name = ctype_to_input.get(k, k)
                        if name in control_inputs:
                            kwargs[name] = v
                else:
                    kwargs["control_image"] = val  # single image (backward compat)

        # ── Explicitly clear optional control/ip inputs when not provided (no stale cache / "sticky" image) ──
        for opt in list(graph_input_names):
            if opt == "ip_image":
                if opt not in kwargs:
                    kwargs[opt] = None
                elif kwargs[opt] is None or kwargs[opt] == [] or kwargs[opt] == {}:
                    kwargs[opt] = None
                elif isinstance(kwargs[opt], dict) and not kwargs[opt].get("image") and not kwargs[opt].get("images"):
                    kwargs[opt] = None
            elif opt == "control_image" or opt.startswith("control_image_"):
                if opt not in kwargs:
                    kwargs[opt] = None

        # ── Resolve image inputs from URL or path (control_image*, t2i_control_image, ip_image, source_image) ──
        control_keys = {k for k in kwargs if k == "control_image" or k.startswith("control_image_")}
        for key in list(control_keys) + ["t2i_control_image", "ip_image", "source_image"]:
            if key not in kwargs:
                continue
            val = kwargs[key]
            if val is None:
                continue
            if key == "ip_image":
                # Full multi-image support: single, list, or dict (vocabulary)
                images = _normalize_ip_image(val)
                if not images:
                    kwargs[key] = None
                    continue
                if len(images) == 1:
                    kwargs[key] = {"image": images[0]}
                else:
                    kwargs[key] = {"images": images}
                continue
            if not isinstance(val, str):
                continue
            pil_img = load_image_from_url_or_path(val)
            if pil_img is None:
                import warnings
                warnings.warn(f"Failed to load image from {val!r}; passing None for {key}.", UserWarning)
                kwargs[key] = None
                continue
            t = _pil_to_tensor(pil_img)
            if t.shape[2] != height or t.shape[3] != width:
                t = torch.nn.functional.interpolate(
                    t, size=(height, width), mode="bilinear", align_corners=False
                )
            if self.graph._device is not None and hasattr(t, "to"):
                t = t.to(self.graph._device)
            kwargs[key] = t

        # ── Use scheduler's timesteps in the loop (diffusers parity: Euler, PNDM, SD3 flow, etc.)
        if "timesteps" not in kwargs and "timesteps" in graph_input_names:
            num_steps = kwargs.get("num_steps") or meta.get("default_num_steps", 50)
            for _, block in self.graph._iter_all_blocks():
                if hasattr(block, "num_iterations"):
                    num_steps = block.num_iterations
                    break
            device = self.graph._device or torch.device("cpu")
            use_scheduler = meta.get("use_euler_init_sigma") or meta.get("use_scheduler_timesteps")
            if use_scheduler:
                for _, block in self.graph._iter_all_blocks():
                    if hasattr(block, "set_timesteps") and hasattr(block, "scheduler"):
                        block.set_timesteps(num_steps, device)
                        kwargs["timesteps"] = block.scheduler.timesteps.clone().to(device=device)
                        break
            elif meta.get("base_model") == "sd3":
                # SD3: use FlowMatchEulerDiscreteScheduler from model repo (shift=3.0, same as Diffusers)
                try:
                    from diffusers import FlowMatchEulerDiscreteScheduler
                    pretrained = meta.get("pretrained", "stabilityai/stable-diffusion-3-medium-diffusers")
                    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained, subfolder="scheduler")
                    sched.set_timesteps(num_steps, device=device)
                    kwargs["timesteps"] = sched.timesteps.clone().to(device=device)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        "SD3: could not set FlowMatchEulerDiscreteScheduler timesteps: %s. Loop will use default.", e
                    )

        # ── Generate noise or I2V/V2V initial latents (only if graph expects latents and none provided) ──
        if "latents" in graph_input_names and "latents" not in kwargs:
            if meta.get("modality") == "audio":
                kwargs["latents"] = self._make_noise_audio(
                    batch_size=batch_size, seed=seed, meta=meta,
                )
            elif meta.get("modality") == "video":
                num_frames = kwargs.get("num_frames") or meta.get("num_frames", 16)
                strength = kwargs.get("strength")
                source_image = kwargs.get("source_image")
                source_video = kwargs.get("source_video")
                if (source_image is not None or source_video is not None):
                    strength = strength if strength is not None else 0.7
                    init_latents, init_timesteps = self._make_initial_latents_from_image_or_video(
                        source_image=source_image, source_video=source_video,
                        width=width, height=height, num_frames=num_frames, strength=strength, seed=seed,
                    )
                    if init_latents is not None:
                        kwargs["latents"] = init_latents
                        if init_timesteps is not None:
                            kwargs["timesteps"] = init_timesteps
                if "latents" not in kwargs:
                    kwargs["latents"] = self._make_noise_video(
                        batch_size=batch_size, num_frames=num_frames, width=width, height=height, seed=seed, meta=meta,
                    )
            else:
                kwargs["latents"] = self._make_noise(
                    batch_size=batch_size, width=width, height=height, seed=seed,
                )
        
        # ── IP-Adapter: включить только если передан ip_image; иначе — голый пайплайн или только ControlNet
        use_ip = bool(kwargs.get("ip_image"))
        self._apply_ip_adapter_switch(use_ip=use_ip)

        # ── IP-Adapter strength: only when ip_image is used; otherwise scale must not affect generation ──
        ip_adapter_scale = kwargs.pop("ip_adapter_scale", None)
        if use_ip and ip_adapter_scale is not None:
            from yggdrasil.core.graph.adapters import set_ip_adapter_scale_on_unet
            for _n, block in self.graph._iter_all_blocks():
                unet = getattr(block, "unet", None)
                if unet is not None and getattr(unet, "attn_processors", None):
                    set_ip_adapter_scale_on_unet(unet, float(ip_adapter_scale))
                    break

        # ── Сила ControlNet: одно значение для всех, список по порядку или словарь по control_type / имени ──
        controlnet_scale = kwargs.pop("controlnet_scale", None) or kwargs.pop("conditioning_scale", None)
        if controlnet_scale is not None:
            _controlnet_blocks_ordered = self._get_controlnet_blocks_ordered()
            if not _controlnet_blocks_ordered:
                pass
            elif isinstance(controlnet_scale, (list, tuple)):
                for i, (_name, block) in enumerate(_controlnet_blocks_ordered):
                    if i < len(controlnet_scale):
                        block.conditioning_scale = float(controlnet_scale[i])
            elif isinstance(controlnet_scale, dict):
                for _name, block in _controlnet_blocks_ordered:
                    ct = getattr(block, "control_type", None)
                    scale = controlnet_scale.get(ct) if ct is not None else None
                    if scale is None:
                        scale = controlnet_scale.get(_name)
                    if scale is None and ct is not None:
                        scale = controlnet_scale.get(f"controlnet-{ct}")
                    if scale is not None:
                        block.conditioning_scale = float(scale)
            else:
                for _name, block in _controlnet_blocks_ordered:
                    block.conditioning_scale = float(controlnet_scale)

        # ── Execute graph (non-strict: tolerate missing optional inputs) ──
        if self._is_flat_diffusers_graph():
            raw = self._run_flat_diffusers_loop(**kwargs)
        else:
            from yggdrasil.core.graph.executor import GraphExecutor
            raw = GraphExecutor(strict=False).execute(self.graph, **kwargs)
        
        # ── Build output ──
        return self._build_output(raw)
    
    def to(self, device=None, dtype=None) -> "InferencePipeline":
        """Move pipeline to device."""
        self.graph.to(device, dtype)
        return self

    def load_lora_weights(
        self,
        pretrained_model_name_or_path: str,
        *,
        weight_name: Optional[str] = None,
        adapter_name: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """Load LoRA weights from HuggingFace (e.g. OnMoon/loras) into the graph's UNet.

        Requires: peft. Uses diffusers-compatible LoRA format.

        Args:
            pretrained_model_name_or_path: HF model id or local path.
            weight_name: Optional .safetensors filename in the repo.
            adapter_name: Adapter name for multi-LoRA.
            **kwargs: Passed to loader (cache_dir, token, revision, etc.).

        Returns:
            List of loaded components (e.g. ["unet"]).
        """
        from yggdrasil.integration.lora_loader import load_lora_weights as _load_lora
        return _load_lora(
            self.graph,
            pretrained_model_name_or_path,
            weight_name=weight_name,
            adapter_name=adapter_name,
            **kwargs,
        )

    # ── Internal helpers ──
    
    def _apply_guidance_scale(self, scale: float):
        for _, block in self.graph._iter_all_blocks():
            bt = getattr(block, 'block_type', '')
            if hasattr(block, 'scale'):
                if 'guidance' in bt or 'batched_cfg' in bt:
                    block.scale = scale
    
    def _apply_num_steps(self, num_steps: int):
        for _, block in self.graph._iter_all_blocks():
            if hasattr(block, 'num_iterations'):
                block.num_iterations = num_steps

    def _apply_ip_adapter_switch(self, use_ip: bool) -> None:
        """Включить или выключить IP-Adapter на UNet по наличию ip_image.

        use_ip=True  — ставим процессоры IP-Adapter (пайплайн с референсным изображением).
        use_ip=False — восстанавливаем обычные процессоры (голый пайплайн / только ControlNet).
        Если при материализации UNet был на meta, процессоры инициализируются при первом use_ip=True.
        """
        from yggdrasil.core.graph.adapters import (
            ensure_ip_adapter_processors_on_device,
            set_ip_adapter_processors_on_unet,
        )
        for _n, block in self.graph._iter_all_blocks():
            unet = getattr(block, "unet", None)
            if unet is None or not hasattr(unet, "set_attn_processor"):
                continue
            has_original = getattr(block, "_ip_adapter_original_processors", None) is not None
            has_scale = getattr(block, "_ip_adapter_scale", None) is not None
            if not has_original and not has_scale:
                continue
            n_attn = len(getattr(unet, "attn_processors", {}))
            if use_ip:
                procs = getattr(block, "_ip_adapter_processors", None)
                if procs is not None and n_attn > 0 and len(procs) == n_attn:
                    unet.set_attn_processor(procs)
                    ensure_ip_adapter_processors_on_device(unet)
                elif n_attn > 0 and has_scale:
                    # Отложенная инициализация: UNet при материализации был на meta
                    scale = float(getattr(block, "_ip_adapter_scale", 0.6))
                    block._ip_adapter_original_processors = dict(unet.attn_processors)
                    set_ip_adapter_processors_on_unet(unet, scale=scale)
                    block._ip_adapter_processors = dict(unet.attn_processors)
                    ensure_ip_adapter_processors_on_device(unet)
            else:
                if has_original and n_attn > 0 and len(block._ip_adapter_original_processors) == n_attn:
                    unet.set_attn_processor(block._ip_adapter_original_processors)

    def _get_denoise_loop_node(self) -> Tuple[Optional[str], Any]:
        """Return (loop_node_name, loop_block) for the graph's denoise loop (by name 'denoise_loop' or by role loop/)."""
        nodes = self.graph.nodes
        loop = nodes.get("denoise_loop")
        if loop is not None:
            return "denoise_loop", loop
        for name, block in nodes.items():
            if getattr(block, "block_type", "").startswith("loop/"):
                return name, block
            if hasattr(block, "_loop") and getattr(block, "_loop", None) is not None:
                return name, block
        return None, None

    def _get_controlnet_input_names_and_types(self) -> Tuple[List[str], Dict[str, str]]:
        """Return (ordered control_image* graph input names, control_type -> input_name map) for multiple ControlNets (§11.3 A2)."""
        graph_input_names = list(self.graph.graph_inputs.keys())
        control_inputs = [k for k in graph_input_names if k == "control_image" or k.startswith("control_image_")]
        def _sort_key(k: str):
            if k == "control_image":
                return (0, 0)
            suffix = k.replace("control_image_", "")
            if suffix.isdigit():
                return (1, int(suffix))
            return (1, 0)
        control_inputs.sort(key=_sort_key)
        # Единый источник: graph.metadata["controlnet_input_mapping"] (заполняется при materialize/infer_metadata)
        ctype_to_input: Dict[str, str] = dict(self.graph.metadata.get("controlnet_input_mapping") or {})
        if not ctype_to_input and control_inputs:
            from yggdrasil.core.graph.adapters import get_controlnet_input_mapping
            ctype_to_input = get_controlnet_input_mapping(self.graph)
        return control_inputs, ctype_to_input

    def _get_controlnet_blocks_ordered(self) -> List[Tuple[str, Any]]:
        """Возвращает список (имя_узла, блок) для всех ControlNet во внутреннем графе цикла.
        Учитываются узлы controlnet, controlnet_1, controlnet_2, а также controlnet-depth, controlnet-canny и т.п."""
        result: List[Tuple[str, Any]] = []
        _loop_name, loop = self._get_denoise_loop_node()
        if loop is None:
            return result
        inner = getattr(loop, "graph", None)
        if inner is None and hasattr(loop, "_loop") and getattr(loop, "_loop", None) is not None:
            inner = getattr(loop._loop, "graph", None)
        if inner is None:
            return result
        # Все узлы с block_type adapter/controlnet (в т.ч. controlnet-depth, controlnet-canny)
        candidates = [
            (n, inner.nodes[n]) for n in inner.nodes
            if getattr(inner.nodes[n], "block_type", None) == "adapter/controlnet"
        ]
        if not candidates:
            return result
        # Сортируем: "controlnet" первый, затем controlnet_1, controlnet_2, затем по имени (controlnet-canny, controlnet-depth)
        prefix = "controlnet"
        def _order_key(item):
            name = item[0]
            if name == prefix:
                return (0, 0, "")
            if name.startswith(prefix + "_") and name[len(prefix) + 1:].isdigit():
                return (1, int(name[len(prefix) + 1:]), "")
            return (2, 0, name)
        candidates.sort(key=_order_key)
        return candidates

    def _is_flat_diffusers_graph(self) -> bool:
        """True if graph is a flat diffusers-imported graph (backbone + solver, no LoopSubGraph)."""
        loop_name, _ = self._get_denoise_loop_node()
        if loop_name is not None:
            return False
        if "solver" not in self.graph.nodes or "backbone" not in self.graph.nodes:
            return False
        outs = getattr(self.graph, "graph_outputs", None) or {}
        return "next_latents" in outs or "decoded" in outs

    def _run_flat_diffusers_loop(self, **kwargs: Any) -> Dict[str, Any]:
        """Run denoising loop for a flat diffusers-style graph (one step per execute)."""
        from yggdrasil.core.graph.executor import GraphExecutor
        executor = GraphExecutor(strict=False, no_grad=True)
        graph = self.graph
        device = graph._device or torch.device("cpu")
        num_steps = int(kwargs.pop("num_steps", None) or graph.metadata.get("default_num_steps", 28))
        latents = kwargs.get("latents")
        if latents is None:
            raise ValueError("flat diffusers graph requires latents (set by InferencePipeline)")
        # Set timesteps on solver if available (Euler, PNDM, etc.)
        solver = graph.nodes.get("solver")
        if solver is not None and hasattr(solver, "set_timesteps"):
            solver.set_timesteps(num_steps, device)
        # Build timesteps (leading spacing like diffusers)
        num_train_t = int(getattr(solver, "num_train_timesteps", None) or graph.metadata.get("num_train_timesteps", 1000))
        steps_offset = int(getattr(solver, "steps_offset", None) or 0)
        step_ratio = num_train_t // num_steps
        timesteps = torch.arange(0, num_steps, device=device).long() * step_ratio
        timesteps = timesteps.flip(0)
        timesteps = (timesteps + steps_offset).clamp(0, num_train_t - 1)
        out: Dict[str, Any] = {}
        for i in range(len(timesteps)):
            t = timesteps[i].unsqueeze(0) if timesteps[i].dim() == 0 else timesteps[i]
            next_t = timesteps[i + 1].unsqueeze(0) if i + 1 < len(timesteps) else torch.tensor(0, device=device)
            if next_t.dim() == 0:
                next_t = next_t.unsqueeze(0)
            step_kw = {**kwargs, "latents": latents, "timestep": t}
            if "next_timestep" in graph.graph_inputs:
                step_kw["next_timestep"] = next_t
            if i == 0 and "num_steps" in graph.graph_inputs:
                step_kw["num_steps"] = num_steps
            out = executor.execute(graph, **step_kw)
            latents = out.get("next_latents")
            if latents is None:
                latents = out.get("latents")
            if latents is None:
                latents = out.get("output")
            if latents is None:
                break
        return out
    
    def _make_noise(self, batch_size=1, width=512, height=512, seed=None):
        meta = self.graph.metadata
        channels = meta.get("latent_channels", 4)
        scale = meta.get("spatial_scale_factor", 8)
        h, w = height // scale, width // scale

        device = self.graph._device or torch.device("cpu")
        device_type = device.type if hasattr(device, "type") else str(device)

        init_sigma = meta.get("init_noise_sigma")
        if init_sigma is None and meta.get("use_euler_init_sigma"):
            num_steps = meta.get("default_num_steps", 50)
            for _, block in self.graph._iter_all_blocks():
                if hasattr(block, "num_iterations"):
                    num_steps = block.num_iterations
                    break
            for _, block in self.graph._iter_all_blocks():
                if getattr(block, "block_type", "") == "solver/euler_discrete":
                    block.set_timesteps(num_steps, device)
                    init_sigma = block.init_noise_sigma
                    break
        if init_sigma is None:
            init_sigma = 1.0

        # Match Diffusers: SD3 and SDXL on cuda use pipeline dtype float16 for prepare_latents (same randn sequence)
        if meta.get("base_model") in ("sd3", "sdxl") and device_type == "cuda":
            dtype = torch.float16
        elif device_type == "mps":
            dtype = torch.float32
        else:
            dtype = torch.get_default_dtype()

        shape = (batch_size, channels, h, w)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device if device_type != "mps" else "cpu").manual_seed(seed)
        # SDXL: use diffusers.randn_tensor for exact parity with prepare_latents
        if meta.get("base_model") == "sdxl":
            try:
                from diffusers.utils.torch_utils import randn_tensor
                noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            except ImportError:
                if generator is not None:
                    noise = torch.randn(shape, device=device, dtype=dtype, generator=generator)
                else:
                    noise = torch.randn(shape, device=device, dtype=dtype)
        else:
            if generator is not None:
                if device_type == "mps":
                    g = torch.Generator().manual_seed(seed)
                    noise = torch.randn(shape, dtype=dtype, generator=g).to(device=device)
                else:
                    noise = torch.randn(shape, device=device, dtype=dtype, generator=generator)
            else:
                noise = torch.randn(shape, device=device, dtype=dtype)

        if init_sigma != 1.0:
            noise = noise * init_sigma

        return noise.to(device=device, dtype=dtype)

    def _make_noise_video(self, batch_size=1, num_frames=16, width=512, height=512, seed=None, meta=None):
        """Initial noise for video pipelines: (batch, channels, num_frames, H, W)."""
        meta = meta or self.graph.metadata
        channels = int(meta.get("latent_channels", 4))
        scale = int(meta.get("spatial_scale_factor", 8))
        h, w = height // scale, width // scale
        device = self.graph._device or torch.device("cpu")
        dtype = torch.get_default_dtype()
        init_sigma = meta.get("init_noise_sigma", 1.0)
        if seed is not None:
            g = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(batch_size, channels, int(num_frames), h, w, device=device, dtype=dtype, generator=g)
        else:
            noise = torch.randn(batch_size, channels, int(num_frames), h, w, device=device, dtype=dtype)
        if init_sigma != 1.0:
            noise = noise * init_sigma
        return noise

    def _make_initial_latents_from_image_or_video(
        self,
        source_image=None,
        source_video=None,
        width=512,
        height=512,
        num_frames=16,
        strength=0.7,
        seed=None,
    ):
        """I2V/V2V: начальные латенты из изображения или видео + шум по strength. Возвращает (latents, timesteps или None)."""
        if source_image is None and source_video is None:
            return None, None
        graph = self.graph
        codec = graph.nodes.get("codec")
        _loop_name, loop = self._get_denoise_loop_node()
        if not codec or not getattr(codec, "encode", None) or not loop:
            return None, None
        inner = getattr(loop, "graph", None) or (getattr(loop, "_loop", None) and getattr(loop._loop, "graph", None))
        if not inner:
            return None, None
        solver = inner.nodes.get("solver") if inner else None
        if not solver or not getattr(solver, "alphas_cumprod", None):
            return None, None
        dev = graph._device or torch.device("cpu")
        meta = graph.metadata
        scale = int(meta.get("spatial_scale_factor", 8))
        h, w = height // scale, width // scale
        T = int(getattr(loop, "num_train_timesteps", 1000))
        num_steps = meta.get("default_num_steps", 25)
        for _, block in graph._iter_all_blocks():
            if hasattr(block, "num_iterations"):
                num_steps = block.num_iterations
                break
        steps_offset = int(getattr(loop, "steps_offset", 0))
        step_ratio = T // num_steps
        timesteps_full = torch.arange(0, num_steps, device=dev).long() * step_ratio
        timesteps_full = timesteps_full.flip(0)
        timesteps_full = (timesteps_full + steps_offset).clamp(0, T - 1)
        start_idx = min(int((1.0 - strength) * num_steps), num_steps - 1)
        start_idx = max(0, start_idx)
        timesteps_slice = timesteps_full[start_idx:]
        first_t = timesteps_slice[0].item()

        if source_video is not None and hasattr(source_video, "shape") and source_video.dim() >= 4:
            with torch.no_grad():
                if source_video.dim() == 4:
                    source_video = source_video.unsqueeze(0)
                B, C, T_in, H, W = source_video.shape
                pixel = source_video.to(dev).float()
                if pixel.max() > 1.0:
                    pixel = pixel / 255.0
                if pixel.min() >= 0 and pixel.max() <= 1.0:
                    pixel = pixel * 2.0 - 1.0
                if T_in != num_frames or H != height or W != width:
                    pixel = torch.nn.functional.interpolate(
                        pixel.reshape(B * T_in, C, H, W), size=(height, width), mode="bilinear", align_corners=False,
                    )
                    pixel = pixel.reshape(B, T_in, C, height, width).permute(0, 2, 1, 3, 4)
                B, C, T_in, H, W = pixel.shape
                pixel_2d = pixel.permute(0, 2, 1, 3, 4).reshape(B * T_in, C, H, W)
                lat_2d = codec.encode(pixel_2d)
                video_latent = lat_2d.reshape(B, T_in, *lat_2d.shape[1:]).permute(0, 2, 1, 3, 4)
                if T_in != num_frames:
                    video_latent = torch.nn.functional.interpolate(
                        video_latent.float(), size=(num_frames, h, w), mode="trilinear", align_corners=False,
                    )
        else:
            if source_image is None:
                return None, None
            pixel = source_image.to(dev) if hasattr(source_image, "shape") else source_image
            if pixel.dim() == 3:
                pixel = pixel.unsqueeze(0)
            if pixel.shape[2] != height or pixel.shape[3] != width:
                pixel = torch.nn.functional.interpolate(
                    pixel.float(), size=(height, width), mode="bilinear", align_corners=False,
                )
            if pixel.min() >= 0 and pixel.max() <= 1.0:
                pixel = pixel * 2.0 - 1.0
            with torch.no_grad():
                image_latent = codec.encode(pixel)
            video_latent = image_latent.unsqueeze(2).expand(1, image_latent.shape[1], num_frames, h, w).clone()

        alpha = solver.alphas_cumprod[first_t].to(device=dev, dtype=video_latent.dtype)
        while alpha.dim() < video_latent.dim():
            alpha = alpha.unsqueeze(-1)
        g = torch.Generator(device=dev).manual_seed(seed) if seed is not None else None
        noise = torch.randn_like(video_latent, device=dev, generator=g)
        noisy = alpha.sqrt() * video_latent + (1 - alpha).sqrt() * noise
        return noisy, timesteps_slice

    def _make_noise_audio(self, batch_size=1, seed=None, meta=None):
        """Initial noise for audio pipelines: (batch, latent_channels, H, W) from metadata."""
        meta = meta or self.graph.metadata
        channels = int(meta.get("latent_channels", 8))
        h = int(meta.get("default_audio_latent_height", 256))
        w = int(meta.get("default_audio_latent_width", 16))
        # Ensure scalars so torch.randn gets tuple of ints, not list (PyTorch requirement)
        batch_size = int(batch_size)
        device = self.graph._device or torch.device("cpu")
        dtype = torch.get_default_dtype()
        shape = (batch_size, channels, h, w)
        if seed is not None:
            g = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(*shape, device=device, dtype=dtype, generator=g)
        else:
            noise = torch.randn(*shape, device=device, dtype=dtype)
        sigma = meta.get("init_noise_sigma", 1.0)
        if sigma != 1.0:
            noise = noise * sigma
        return noise

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
            elif decoded.dim() == 3 or decoded.dim() <= 2:
                # Audio: (batch, channels, time) or (batch, time)
                output.audio = decoded
        
        # If no decoded, check for raw output (output, image — §11.5 N2 non-diffusion contract)
        if output.images is None and output.audio is None and output.video is None:
            out_tensor = raw.get("output") or raw.get("image")
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
        # VAE outputs [-1, 1]; denormalize to [0, 1] for PIL
        if img.min() < -0.01 or img.max() > 1.01:
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
        return f"<InferencePipeline '{self.graph.name}' device={device} nodes={len(self.graph.nodes)}>"


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


class TrainingPipeline:
    """High-level train API: train a ComputeGraph (subset of nodes or full graph).

    Example::
        train_pipe = TrainingPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", train_nodes=["lora_adapter"])
        train_pipe.train(data_path="./data", epochs=10, lr=1e-4)
        train_pipe.save_checkpoint("checkpoints/sd15_lora")
    """

    def __init__(self, graph, *, train_nodes=None, freeze_nodes=None, train_stages=None, device=None, **config):
        from yggdrasil.core.graph.graph import ComputeGraph
        self.graph: ComputeGraph = graph
        self.train_nodes = train_nodes or []
        self.freeze_nodes = freeze_nodes  # T2: if set, train all nodes except these
        self.train_stages = train_stages  # for multi-stage; reserved
        self._config = config
        if device is not None:
            self.graph.to(device)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        train_nodes=None,
        freeze_nodes=None,
        train_stages=None,
        device=None,
        **kwargs,
    ) -> "TrainingPipeline":
        """Build training pipeline from HuggingFace model ID or template."""
        from yggdrasil.integration.diffusers import DiffusersBridge
        try:
            graph = DiffusersBridge.from_pretrained(model_id, **kwargs)
        except Exception:
            from yggdrasil.core.graph.graph import ComputeGraph
            from yggdrasil.hub import resolve_model
            template_name, template_kwargs = resolve_model(model_id)
            graph = ComputeGraph.from_template(template_name, **template_kwargs)
        return cls(graph, train_nodes=train_nodes, freeze_nodes=freeze_nodes, train_stages=train_stages, device=device)

    @classmethod
    def from_template(
        cls,
        template_name: str,
        *,
        train_nodes=None,
        freeze_nodes=None,
        train_stages=None,
        device=None,
        **kwargs,
    ) -> "TrainingPipeline":
        """Build training pipeline from a named template (TZ §6.2).

        Supports inference templates (e.g. sd15_txt2img, sdxl_txt2img) and training
        templates (e.g. train_lora_sd15, train_controlnet). When train_nodes is not
        set and the template has metadata default_train_nodes, those are used.

        Example::
            train_pipe = TrainingPipeline.from_template("train_lora_sd15", device="cuda")
            train_pipe.train(data_path="./data", epochs=10)
        """
        from yggdrasil.core.graph.graph import ComputeGraph
        graph = ComputeGraph.from_template(template_name, **kwargs)
        if train_nodes is None and isinstance(getattr(graph, "metadata", None), dict):
            default = graph.metadata.get("default_train_nodes")
            if default is not None:
                train_nodes = default if isinstance(default, (list, tuple)) else [default]
        return cls(graph, train_nodes=train_nodes, freeze_nodes=freeze_nodes, train_stages=train_stages, device=device)

    @classmethod
    def from_config(cls, path: str, *, train_nodes=None, freeze_nodes=None, train_stages=None, device=None, **overrides) -> "TrainingPipeline":
        """Build training pipeline from YAML/JSON config."""
        from yggdrasil.core.graph.graph import ComputeGraph
        graph = ComputeGraph.from_yaml(path)
        return cls(graph, train_nodes=train_nodes, freeze_nodes=freeze_nodes, train_stages=train_stages, device=device, **overrides)

    @classmethod
    def from_graph(cls, graph, *, train_nodes=None, freeze_nodes=None, train_stages=None, device=None, **config) -> "TrainingPipeline":
        """Build training pipeline from an existing ComputeGraph."""
        return cls(graph, train_nodes=train_nodes, freeze_nodes=freeze_nodes, train_stages=train_stages, device=device, **config)

    @classmethod
    def from_spec(
        cls,
        spec: Union[Any, str, Path],
        *,
        train_nodes=None,
        freeze_nodes=None,
        train_stages=None,
        device=None,
        **kwargs,
    ) -> "TrainingPipeline":
        """Unified entry point: create training pipeline from graph or config path (§11.7 P4).

        Dispatches by type: ComputeGraph -> from_graph(spec); str | Path -> from_config(spec).
        """
        from yggdrasil.core.graph.graph import ComputeGraph
        if isinstance(spec, ComputeGraph):
            return cls.from_graph(spec, train_nodes=train_nodes, freeze_nodes=freeze_nodes, train_stages=train_stages, device=device, **kwargs)
        if isinstance(spec, (str, Path)):
            return cls.from_config(str(spec), train_nodes=train_nodes, freeze_nodes=freeze_nodes, train_stages=train_stages, device=device, **kwargs)
        raise TypeError(
            f"TrainingPipeline.from_spec(spec) expects ComputeGraph or path; got {type(spec).__name__}"
        )

    def train(
        self,
        data_path: str = None,
        dataset=None,
        *,
        epochs: int = 10,
        lr: float = 1e-4,
        batch_size: int = 1,
        **kwargs,
    ):
        """Run training. Uses GraphTrainer with self.graph and self.train_nodes (or train_stages)."""
        from yggdrasil.training.graph_trainer import GraphTrainer, GraphTrainingConfig
        from yggdrasil.core.graph.stage import AbstractStage

        if dataset is None and data_path:
            from yggdrasil.training.data import ImageFolderSource
            dataset = ImageFolderSource(data_path)
        if dataset is None:
            raise ValueError("Provide data_path=... or dataset=...")
        cfg = GraphTrainingConfig(
            num_epochs=epochs,
            learning_rate=lr,
            batch_size=batch_size,
            **{k: v for k, v in (self._config or {}).items() if k in GraphTrainingConfig.__dataclass_fields__},
        )
        train_nodes = self.train_nodes
        if self.train_stages is not None and not train_nodes:
            # Combined pipeline: train_stages = [0, 1] -> train first two stage nodes
            names = list(self.graph.nodes.keys())
            train_nodes = [names[i] for i in self.train_stages if 0 <= i < len(names)]
        if not train_nodes and self.freeze_nodes is None:
            train_nodes = list(self.graph.nodes)
        if self.freeze_nodes is not None:
            trainer = GraphTrainer(graph=self.graph, freeze_nodes=self.freeze_nodes, config=cfg)
        else:
            trainer = GraphTrainer(graph=self.graph, train_nodes=train_nodes or [], config=cfg)
        trainer.train(dataset, **kwargs)
        self._trainer = trainer
        return self

    def save_checkpoint(self, path: str):
        """Save checkpoint (trainable nodes or full graph)."""
        if hasattr(self, "_trainer") and self._trainer is not None:
            self._trainer.save_checkpoint(path)
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.graph.to_yaml(Path(path) / "graph.yaml")
