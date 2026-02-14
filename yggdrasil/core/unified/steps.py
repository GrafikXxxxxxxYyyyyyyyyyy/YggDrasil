# yggdrasil/core/unified/steps.py
"""Канонические шаги любого диффузионного процесса.

Любой пайплайн (любая модальность и архитектура) раскладывается на эти абстрактные шаги.
Конкретные блоки (UNet, DiT, VAE, Encodec, …) реализуют эти шаги через единые порты.
"""
from __future__ import annotations

from enum import Enum
from typing import List


class DiffusionStep(str, Enum):
    """Абстрактные шаги диффузионного пайплайна (модальность-независимые)."""
    # Вход в латентное пространство
    ENCODE = "encode"                     # raw_signal -> initial_latents (codec / init noise)
    # Условие
    CONDITION = "condition"               # prompt / class / etc -> condition embedding
    # Управление генерацией (ControlNet, T2I-Adapter, etc.)
    CONTROL = "control"                   # control_signal -> adapter_features (опционально)
    # Один шаг обратного процесса (денойзинг)
    REVERSE_STEP = "reverse_step"         # latents, timestep, condition, [adapter_features] -> next_latents
    # Цикл денойзинга = последовательность REVERSE_STEP
    DENOISE_LOOP = "denoise_loop"
    # Выход из латентного пространства
    DECODE = "decode"                     # latents -> output_signal (codec)
    # Улучшение результата (FaceDetailer, upscaler, etc.)
    POST_PROCESS = "post_process"          # output_signal -> enhanced_signal (опционально)

    # Для обучения
    FORWARD_PROCESS = "forward_process"   # x0, t, noise -> xt, target
    LOSS = "loss"                         # prediction, target -> loss


# Шаги, которые обычно присутствуют в inference-графе (порядок логический)
CANONICAL_INFERENCE_STEPS: List[DiffusionStep] = [
    DiffusionStep.CONDITION,
    DiffusionStep.CONTROL,
    DiffusionStep.ENCODE,
    DiffusionStep.DENOISE_LOOP,
    DiffusionStep.DECODE,
    DiffusionStep.POST_PROCESS,
]


# Шаги для training-графа
CANONICAL_TRAINING_STEPS: List[DiffusionStep] = [
    DiffusionStep.CONDITION,
    DiffusionStep.FORWARD_PROCESS,
    DiffusionStep.REVERSE_STEP,  # backbone prediction
    DiffusionStep.LOSS,
]
