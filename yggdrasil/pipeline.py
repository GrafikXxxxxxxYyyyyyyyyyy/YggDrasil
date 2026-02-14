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

import io
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
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
        
        # 3. Fallback: guess template from model_id (preset names or HF-style ids)
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
                for k, v in Pipeline.list_available().items()
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
                for k, v in Pipeline.list_available().items()
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

        # ── Use scheduler's timesteps in the loop (diffusers parity: Euler, PNDM, etc.)
        if "timesteps" not in kwargs and "timesteps" in graph_input_names:
            use_scheduler = meta.get("use_euler_init_sigma") or meta.get("use_scheduler_timesteps")
            if use_scheduler:
                num_steps = meta.get("default_num_steps", 50)
                for _, block in self.graph._iter_all_blocks():
                    if hasattr(block, "num_iterations"):
                        num_steps = block.num_iterations
                        break
                device = self.graph._device or torch.device("cpu")
                for _, block in self.graph._iter_all_blocks():
                    if hasattr(block, "set_timesteps") and hasattr(block, "scheduler"):
                        block.set_timesteps(num_steps, device)
                        kwargs["timesteps"] = block.scheduler.timesteps.clone().to(device=device)
                        break

        # ── Generate noise (only if graph expects latents and none provided) ──
        if "latents" in graph_input_names and "latents" not in kwargs:
            if meta.get("modality") == "audio":
                kwargs["latents"] = self._make_noise_audio(
                    batch_size=batch_size, seed=seed, meta=meta,
                )
            elif meta.get("modality") == "video":
                num_frames = kwargs.get("num_frames") or meta.get("num_frames", 16)
                kwargs["latents"] = self._make_noise_video(
                    batch_size=batch_size, num_frames=num_frames, width=width, height=height, seed=seed, meta=meta,
                )
            else:
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

        # Use float32 for initial noise so the denoising loop accumulates in float32 (avoids banding; diffusers does this on MPS)
        dtype = torch.float32 if device_type == "mps" else torch.get_default_dtype()
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

    def _make_noise_audio(self, batch_size=1, seed=None, meta=None):
        """Initial noise for audio pipelines: (batch, latent_channels, H, W) from metadata."""
        meta = meta or self.graph.metadata
        channels = int(meta.get("latent_channels", 8))
        h = int(meta.get("default_audio_latent_height", 256))
        w = int(meta.get("default_audio_latent_width", 16))
        device = self.graph._device or torch.device("cpu")
        dtype = torch.get_default_dtype()
        if seed is not None:
            g = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(batch_size, channels, h, w, device=device, dtype=dtype, generator=g)
        else:
            noise = torch.randn(batch_size, channels, h, w, device=device, dtype=dtype)
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
