# yggdrasil/core/engine/state.py
"""DiffusionState — контейнер состояния диффузионного процесса.

Это НЕ блок, а чистый data-объект, который передаётся между sampler, loop и callbacks.
"""
from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class DiffusionState:
    """Состояние диффузионного процесса на каждом шаге.
    
    Передаётся между sampler, loop, hooks и callbacks.
    Хранит всё, что нужно для одного шага диффузии любой модальности.
    """
    
    # === Основные поля ===
    latents: torch.Tensor                               # Текущие латенты [B, C, *spatial]
    timestep: torch.Tensor                              # Текущий таймстеп
    condition: Dict[str, Any] = field(default_factory=dict)  # Условия генерации
    
    # === Кэши предыдущего шага ===
    prev_latents: Optional[torch.Tensor] = None         # Латенты на предыдущем шаге
    prev_timestep: Optional[torch.Tensor] = None        # Предыдущий таймстеп
    noise: Optional[torch.Tensor] = None                # Начальный шум (для воспроизводимости)
    model_output: Optional[torch.Tensor] = None         # Выход модели на текущем шаге
    
    # === Метаданные (для любых модальностей и расширений) ===
    metadata: Dict[str, Any] = field(default_factory=dict)
    step_index: int = 0                                 # Номер текущего шага
    total_steps: int = 0                                # Общее число шагов
    
    # === Для multi-scale и каскадных моделей ===
    extra_tensors: Dict[str, torch.Tensor] = field(default_factory=dict)

    def to(self, device: torch.device) -> DiffusionState:
        """Перемещение всех тензоров на устройство."""
        self.latents = self.latents.to(device)
        if isinstance(self.timestep, torch.Tensor):
            self.timestep = self.timestep.to(device)
        if self.prev_latents is not None:
            self.prev_latents = self.prev_latents.to(device)
        if self.prev_timestep is not None and isinstance(self.prev_timestep, torch.Tensor):
            self.prev_timestep = self.prev_timestep.to(device)
        if self.noise is not None:
            self.noise = self.noise.to(device)
        if self.model_output is not None:
            self.model_output = self.model_output.to(device)
        for k, v in self.extra_tensors.items():
            self.extra_tensors[k] = v.to(device)
        return self
    
    def clone(self) -> DiffusionState:
        """Глубокая копия (для безопасного ветвления в хуках)."""
        return DiffusionState(
            latents=self.latents.clone(),
            timestep=self.timestep.clone() if isinstance(self.timestep, torch.Tensor) else self.timestep,
            condition=self.condition.copy(),
            prev_latents=self.prev_latents.clone() if self.prev_latents is not None else None,
            prev_timestep=self.prev_timestep.clone() if self.prev_timestep is not None and isinstance(self.prev_timestep, torch.Tensor) else self.prev_timestep,
            noise=self.noise.clone() if self.noise is not None else None,
            model_output=self.model_output.clone() if self.model_output is not None else None,
            metadata=self.metadata.copy(),
            step_index=self.step_index,
            total_steps=self.total_steps,
            extra_tensors={k: v.clone() for k, v in self.extra_tensors.items()},
        )

    @property
    def progress(self) -> float:
        """Прогресс генерации от 0.0 до 1.0."""
        if self.total_steps <= 0:
            return 0.0
        return min(1.0, self.step_index / self.total_steps)

    def __repr__(self) -> str:
        shape = tuple(self.latents.shape) if self.latents is not None else None
        return f"<DiffusionState step={self.step_index}/{self.total_steps} shape={shape} device={self.latents.device}>"
