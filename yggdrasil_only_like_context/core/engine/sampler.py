# yggdrasil/core/engine/sampler.py
"""DiffusionSampler — graph engine only; no slots. Holds model and optional process/solver/schedule."""
from __future__ import annotations

import torch
from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Optional, Callable, List, Generator
from pathlib import Path

from ...core.block.base import AbstractBaseBlock
from ...core.block.registry import register_block
from ...core.block.port import Port, InputPort, OutputPort, TensorSpec
from ...core.model.modular import ModularDiffusionModel
from ...core.block.builder import BlockBuilder
from ...core.engine.state import DiffusionState


@register_block("engine/sampler")
class DiffusionSampler(AbstractBaseBlock):
    """Universal diffusion sampler — graph engine only (no slots).

    Holds: model (required), optional diffusion_process, solver, noise_schedule
    built from config. Sampling loop: model forward -> solver step -> decode.
    """

    block_type = "engine/sampler"
    block_version = "2.0.0"

    def __init__(self, config: DictConfig | dict, model: Optional[ModularDiffusionModel] = None):
        super().__init__(config)
        self._model: Optional[ModularDiffusionModel] = model
        if self._model is None:
            cfg_model = self.config.get("model")
            if isinstance(cfg_model, ModularDiffusionModel):
                self._model = cfg_model
            elif isinstance(cfg_model, dict) and (cfg_model.get("type") or cfg_model.get("block_type")):
                self._model = BlockBuilder.build(cfg_model)
        self._process = self._build_optional("diffusion_process", {"type": "diffusion/process/rectified_flow"})
        self._solver = self._build_optional("solver", {"type": "diffusion/solver/heun"})
        self._schedule = self._build_optional("noise_schedule", {"type": "noise/schedule/cosine"})

        self.num_inference_steps = self.config.get("num_inference_steps", 50)
        self.guidance_scale = self.config.get("guidance_scale", 7.5)
        self.eta = self.config.get("eta", 0.0)
        self.show_progress = self.config.get("show_progress", True)
        self.pre_step_hooks: List[Callable] = []
        self.post_step_hooks: List[Callable] = []
        self.current_state: Optional[DiffusionState] = None

    def _build_optional(self, key: str, default: dict):
        cfg = self.config.get(key, default)
        if cfg is None:
            return None
        if isinstance(cfg, dict) and (cfg.get("type") or cfg.get("block_type")):
            return BlockBuilder.build(cfg)
        return None

    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        return {
            "condition": InputPort("condition", data_type="dict", description="Generation condition"),
            "shape": InputPort("shape", data_type="any", optional=True, description="Output shape"),
            "num_inference_steps": InputPort("num_inference_steps", data_type="scalar", optional=True),
            "guidance_scale": InputPort("guidance_scale", data_type="scalar", optional=True),
            "result": OutputPort("result", description="Generated output (decoded)"),
            "latents": OutputPort("latents", spec=TensorSpec(space="latent"), description="Final latents"),
        }

    def process(self, **port_inputs) -> Dict[str, Any]:
        condition = port_inputs.get("condition", {})
        shape = port_inputs.get("shape")
        steps = port_inputs.get("num_inference_steps")
        scale = port_inputs.get("guidance_scale")
        result = self.sample(condition=condition, shape=shape, num_inference_steps=steps, guidance_scale=scale)
        return {"result": result, "output": result}

    def _forward_impl(self, condition: Dict[str, Any], shape=None, **kwargs):
        return self.sample(condition=condition, shape=shape, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        condition: Dict[str, Any],
        shape: tuple[int, ...] | None = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        callback: Optional[Callable[[int, torch.Tensor], None]] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("DiffusionSampler: model not set (pass model= or config.model)")
        steps = num_inference_steps or self.num_inference_steps
        scale = guidance_scale or self.guidance_scale
        latents = self._initialize_latents(shape, generator)
        device = latents.device
        timesteps = self._get_timesteps(steps).to(device=device).long()
        state = DiffusionState(latents=latents, timestep=timesteps[0], total_steps=len(timesteps))

        for i in tqdm(range(len(timesteps)), disable=not self.show_progress, desc="Sampling"):
            state.timestep = timesteps[i]
            state.step_index = i
            next_t = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device).long()
            for hook in self.pre_step_hooks:
                hook_result = hook(state, i)
                if hook_result is not None:
                    state = hook_result
            state = self.step(state, condition, guidance_scale=scale, next_timestep=next_t, **kwargs)
            for hook in self.post_step_hooks:
                hook_result = hook(state, i)
                if hook_result is not None:
                    state = hook_result
            if callback is not None:
                callback(i, state.latents)
        return self._model.decode(state.latents)

    def step(
        self,
        state: DiffusionState,
        condition: Dict[str, Any],
        guidance_scale: float = 7.5,
        **kwargs
    ) -> DiffusionState:
        model = self._model
        process = self._process
        solver = self._solver
        model_output = model(x=state.latents, t=state.timestep, condition=condition, return_dict=False)
        def model_fn(latents: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return model(x=latents, t=t, condition=condition, return_dict=False)
        if solver is not None:
            next_latents = solver.step(
                model_output=model_output,
                current_latents=state.latents,
                timestep=state.timestep,
                process=process,
                model_fn=model_fn,
                **kwargs
            )
        elif process is not None:
            next_latents = process.reverse_step(model_output, state.latents, state.timestep)
        else:
            next_latents = state.latents  # no-op
        state.model_output = model_output
        state.prev_latents = state.latents.clone()
        state.latents = next_latents
        state.prev_timestep = state.timestep
        state.step_index += 1
        return state

    def _initialize_latents(
        self,
        shape: tuple[int, ...] | None,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        if shape is None:
            model = self._model
            if hasattr(model, "_graph") and model._graph and "codec" in model._graph.nodes:
                codec = model._graph.nodes["codec"]
                shape = codec.get_latent_shape() if hasattr(codec, "get_latent_shape") else (1, 4, 64, 64)
            else:
                shape = (1, 4, 64, 64)
        device = next(self._model.parameters(), torch.tensor(0)).device
        use_mps = getattr(device, "type", None) == "mps" or (generator and "mps" in str(getattr(generator, "device", "")))
        if use_mps:
            g = torch.Generator().manual_seed(generator.initial_seed()) if generator else None
            noise = torch.randn(shape, generator=g).to(device)
        else:
            noise = torch.randn(shape, device=device, generator=generator)
        return noise

    def _get_timesteps(self, num_steps: int) -> torch.Tensor:
        if self._schedule is not None:
            return self._schedule.get_timesteps(num_steps)
        num_train = getattr(self._process, "num_train_timesteps", 1000) if self._process else 1000
        step_ratio = num_train // num_steps
        timesteps = torch.arange(num_steps) * step_ratio
        return timesteps.flip(0)

    def add_pre_step_hook(self, hook: Callable):
        self.pre_step_hooks.append(hook)

    def add_post_step_hook(self, hook: Callable):
        self.post_step_hooks.append(hook)

    def clear_hooks(self):
        self.pre_step_hooks.clear()
        self.post_step_hooks.clear()

    def sample_iter(self, condition: Dict[str, Any], **kwargs) -> Generator[torch.Tensor, None, None]:
        steps = kwargs.get("num_inference_steps", self.num_inference_steps)
        timesteps = self._get_timesteps(steps)
        latents = self._initialize_latents(kwargs.get("shape"))
        state = DiffusionState(latents=latents, timestep=timesteps[0])
        for t in timesteps:
            state.timestep = t
            state = self.step(state, condition, **kwargs)
            yield self._model.decode(state.latents)

    def save(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config, path / "sampler_config.yaml")

    @classmethod
    def load(cls, path: Path | str, model: ModularDiffusionModel) -> DiffusionSampler:
        path = Path(path)
        config = OmegaConf.load(path / "sampler_config.yaml")
        return cls(config, model=model)
