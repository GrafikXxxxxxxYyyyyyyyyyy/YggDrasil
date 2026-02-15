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
    to the underlying graph.
    """

    def __init__(self, graph, *, device=None, dtype=None):
        from yggdrasil.core.graph.graph import ComputeGraph
        self.graph: ComputeGraph = graph
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
            kwargs["num_steps"] = num_steps  # so flat diffusers loop can use it
        
        # ── Prepare prompt (only if graph expects it) ──
        if prompt is not None:
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

        # ── Resolve image inputs from URL or path (control_image, t2i_control_image, ip_image, source_image) ──
        for key in ("control_image", "t2i_control_image", "ip_image", "source_image"):
            if key not in kwargs:
                continue
            val = kwargs[key]
            if not isinstance(val, str):
                continue
            pil_img = load_image_from_url_or_path(val)
            if pil_img is None:
                import warnings
                warnings.warn(f"Failed to load image from {val!r}; passing None for {key}.", UserWarning)
                kwargs[key] = None
                continue
            if key == "ip_image":
                # IP-Adapter / CLIP vision expect dict with "image" or raw image
                kwargs[key] = {"image": pil_img}
            else:
                # control_image, t2i_control_image, source_image: tensor (1, 3, H, W) in [0, 1], resized to generation size
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
            num_steps = meta.get("default_num_steps", 50)
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

    def _is_flat_diffusers_graph(self) -> bool:
        """True if graph is a flat diffusers-imported graph (backbone + solver, no LoopSubGraph)."""
        if "denoise_loop" in self.graph.nodes:
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

        # Match Diffusers: SD3 on cuda uses pipeline dtype float16 for prepare_latents (same randn sequence)
        if meta.get("base_model") == "sd3" and device_type == "cuda":
            dtype = torch.float16
        elif device_type == "mps":
            dtype = torch.float32
        else:
            dtype = torch.get_default_dtype()
        if seed is not None:
            if device_type == "mps":
                g = torch.Generator().manual_seed(seed)
                noise = torch.randn(batch_size, channels, h, w, dtype=dtype, generator=g)
            else:
                g = torch.Generator(device).manual_seed(seed)
                noise = torch.randn(batch_size, channels, h, w, device=device, dtype=dtype, generator=g)
        else:
            noise = torch.randn(batch_size, channels, h, w, dtype=dtype)

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
        loop = graph.nodes.get("denoise_loop")
        if not codec or not getattr(codec, "encode", None) or not loop or not getattr(loop, "graph", None):
            return None, None
        inner = loop.graph
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

    def __init__(self, graph, *, train_nodes=None, train_stages=None, device=None, **config):
        from yggdrasil.core.graph.graph import ComputeGraph
        self.graph: ComputeGraph = graph
        self.train_nodes = train_nodes or []
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
        return cls(graph, train_nodes=train_nodes, train_stages=train_stages, device=device)

    @classmethod
    def from_template(
        cls,
        template_name: str,
        *,
        train_nodes=None,
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
        return cls(graph, train_nodes=train_nodes, train_stages=train_stages, device=device)

    @classmethod
    def from_config(cls, path: str, *, train_nodes=None, train_stages=None, device=None, **overrides) -> "TrainingPipeline":
        """Build training pipeline from YAML/JSON config."""
        from yggdrasil.core.graph.graph import ComputeGraph
        graph = ComputeGraph.from_yaml(path)
        return cls(graph, train_nodes=train_nodes, train_stages=train_stages, device=device, **overrides)

    @classmethod
    def from_graph(cls, graph, *, train_nodes=None, train_stages=None, device=None, **config) -> "TrainingPipeline":
        """Build training pipeline from an existing ComputeGraph."""
        return cls(graph, train_nodes=train_nodes, train_stages=train_stages, device=device, **config)

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
        if not train_nodes:
            train_nodes = list(self.graph.nodes)
        trainer = GraphTrainer(graph=self.graph, train_nodes=train_nodes, config=cfg)
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
