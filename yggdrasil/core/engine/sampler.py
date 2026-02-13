from __future__ import annotations

import torch
from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Optional, Callable, List, Generator
from pathlib import Path

from ...core.block.base import AbstractBlock
from ...core.block.registry import register_block
from ...core.block.slot import Slot

from ...core.model.modular import ModularDiffusionModel
from ...core.diffusion.process import AbstractDiffusionProcess
from ...core.diffusion.solver import AbstractSolver
from ...core.diffusion.noise.schedule import NoiseSchedule
from ...core.utils.tensor import DiffusionTensor

from ...core.engine.state import DiffusionState


@register_block("engine/sampler")
class DiffusionSampler(AbstractBlock):
    """Универсальный DiffusionSampler — Lego-оркестратор генерации.
    
    Работает с ЛЮБОЙ моделью, ЛЮБОЙ модальностью и ЛЮБЫМ процессом.
    Один и тот же класс используется для:
    - Изображений (SDXL, Flux)
    - Видео (CogVideoX)
    - Аудио (Stable Audio)
    - 3D (Gaussian Splatting)
    - Молекул, временных рядов, текста и твоих кастомных модальностей
    """
    
    block_type = "engine/sampler"
    block_version = "1.0.0"
    
    def __init__(self, config: DictConfig | dict, model: Optional[ModularDiffusionModel] = None):
        super().__init__(config)
        
        # Если модель передана явно — подключаем в слот
        if model is not None:
            self.attach_slot("model", model)
        
        # Настройки по умолчанию
        self.num_inference_steps = self.config.get("num_inference_steps", 50)
        self.guidance_scale = self.config.get("guidance_scale", 7.5)
        self.eta = self.config.get("eta", 0.0)                    # для DDIM
        self.show_progress = self.config.get("show_progress", True)
        
        # Кэш состояния (для streaming и продолжения генерации)
        self.current_state: Optional[DiffusionState] = None
    
    def _define_slots(self) -> Dict[str, Slot]:
        """Lego-дырки сэмплера."""
        return {
            "model": Slot(
                name="model",
                accepts=ModularDiffusionModel,
                multiple=False,
                optional=False
            ),
            "diffusion_process": Slot(
                name="diffusion_process",
                accepts=AbstractDiffusionProcess,
                multiple=False,
                optional=True,
                default={"type": "diffusion/process/rectified_flow"}
            ),
            "solver": Slot(
                name="solver",
                accepts=AbstractSolver,
                multiple=False,
                optional=True,
                default={"type": "diffusion/solver/heun"}
            ),
            "noise_schedule": Slot(
                name="noise_schedule",
                accepts=NoiseSchedule,
                multiple=False,
                optional=True,
                default={"type": "noise/schedule/cosine"}
            ),
        }
    
    def _forward_impl(self, condition: Dict[str, Any], shape=None, **kwargs):
        """Требуется AbstractBlock; делегирует в sample()."""
        return self.sample(condition=condition, shape=shape, **kwargs)

    # ==================== ОСНОВНОЙ ИНТЕРФЕЙС ====================
    
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
        """Главный метод генерации.
        
        Возвращает готовый результат в пространстве данных (после decode).
        """
        steps = num_inference_steps or self.num_inference_steps
        scale = guidance_scale or self.guidance_scale
        
        # 1. Инициализация латентов
        latents = self._initialize_latents(shape, generator)
        device = latents.device
        dtype = latents.dtype

        # 2. Получаем расписание таймстепов (целые числа на device модели)
        timesteps = self._get_timesteps(steps).to(device=device).long()

        # 3. Основной цикл сэмплирования
        state = DiffusionState(latents=latents, timestep=timesteps[0])
        
        for i in tqdm(range(len(timesteps)), disable=not self.show_progress, desc="Sampling"):
            state.timestep = timesteps[i]
            next_t = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device).long()
            
            state = self.step(state, condition, guidance_scale=scale, next_timestep=next_t, **kwargs)
            
            if callback is not None:
                callback(i, state.latents)
        
        # 4. Финальное декодирование
        result = self._slot_children["model"].decode(state.latents)
        return result
    
    def step(
        self,
        state: DiffusionState,
        condition: Dict[str, Any],
        guidance_scale: float = 7.5,
        **kwargs
    ) -> DiffusionState:
        """Один шаг диффузии (используется и в цикле, и для streaming)."""
        model = self._slot_children["model"]
        process = self._slot_children.get("diffusion_process")
        solver = self._slot_children.get("solver")
        
        # 1. Forward модели (с guidance внутри модели)
        model_output = model(
            x=state.latents,
            t=state.timestep,
            condition=condition,
            return_dict=False
        )
        
        # 2. Применяем solver (DDIM, Heun, Euler и т.д.)
        if solver is not None:
            next_latents = solver.step(
                model_output=model_output,
                current_latents=state.latents,
                timestep=state.timestep,
                process=process,
                **kwargs
            )
        else:
            # Fallback — простой процесс
            next_latents = process.reverse_step(
                model_output, state.latents, state.timestep
            )
        
        state.latents = next_latents
        state.prev_timestep = state.timestep
        return state
    
    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================
    
    def _initialize_latents(
        self,
        shape: tuple[int, ...] | None,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Создаём начальный шум."""
        if shape is None:
            # Берём shape из модели (codec)
            codec = self._slot_children["model"]._slot_children.get("codec")
            shape = codec.get_latent_shape() if codec else (1, 4, 64, 64)  # fallback
        
        device = next(self._slot_children["model"].parameters(), torch.tensor(0)).device
        # MPS: torch.randn(..., device=mps, generator=generator) даёт "Placeholder storage has not been allocated".
        # Генерируем шум на CPU (с тем же seed при наличии generator) и переносим на device.
        use_mps_workaround = (
            getattr(device, "type", None) == "mps"
            or (generator is not None and "mps" in str(getattr(generator, "device", "")))
        )
        if use_mps_workaround:
            g = torch.Generator().manual_seed(generator.initial_seed()) if generator is not None else None
            noise = torch.randn(shape, generator=g).to(device)
        else:
            noise = torch.randn(shape, device=device, generator=generator)
        return noise
    
    def _get_timesteps(self, num_steps: int) -> torch.Tensor:
        """Расписание таймстепов (от scheduler или дефолтное для SD 1.5).
        
        Возвращает целочисленные шаги от 999 до ~0, равномерно распределённые.
        UNet SD 1.5 ожидает int-таймстепы в диапазоне [0, 999].
        """
        schedule = self._slot_children.get("noise_schedule")
        if schedule is not None:
            return schedule.get_timesteps(num_steps)
        # Дефолтное расписание для SD 1.5: 999 → 0, num_steps шагов
        process = self._slot_children.get("diffusion_process")
        num_train = getattr(process, "num_train_timesteps", 1000) if process else 1000
        step_ratio = num_train // num_steps
        timesteps = torch.arange(num_steps) * step_ratio
        timesteps = timesteps.flip(0)  # от большого шума к малому: 999, 964, ...
        return timesteps
    
    # ==================== STREAMING И ПРОДОЛЖЕНИЕ ====================
    
    def sample_iter(
        self,
        condition: Dict[str, Any],
        **kwargs
    ) -> Generator[torch.Tensor, None, None]:
        """Генератор для streaming (идеально для Gradio / API)."""
        steps = kwargs.get("num_inference_steps", self.num_inference_steps)
        timesteps = self._get_timesteps(steps)
        latents = self._initialize_latents(kwargs.get("shape"))
        
        state = DiffusionState(latents=latents, timestep=timesteps[0])
        
        for t in timesteps:
            state.timestep = t
            state = self.step(state, condition, **kwargs)
            yield self._slot_children["model"].decode(state.latents)  # промежуточный результат
    
    # ==================== СОХРАНЕНИЕ / ЗАГРУЗКА ====================
    
    def save(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config, path / "sampler_config.yaml")
        # Модель сохраняется отдельно (через model.save())
    
    @classmethod
    def load(cls, path: Path | str, model: ModularDiffusionModel) -> DiffusionSampler:
        path = Path(path)
        config = OmegaConf.load(path / "sampler_config.yaml")
        return cls(config, model=model)