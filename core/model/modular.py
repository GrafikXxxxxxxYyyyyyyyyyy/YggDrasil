from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Any, Dict, Optional, Tuple, List

from ...core.block.base import AbstractBlock
from ...core.block.registry import register_block
from ...core.block.slot import Slot

from .backbone import AbstractBackbone
from .codec import AbstractLatentCodec
from .conditioner import AbstractConditioner
from .guidance import AbstractGuidance
from .position import AbstractPositionEmbedder

from ...core.diffusion.process import AbstractDiffusionProcesss


@register_block("model/modular")
class ModularDiffusionModel(AbstractBlock, nn.Module):
    """Единственная модель во всём YggDrasil.
    
    Это настоящий Lego-конструктор:
    - backbone, codec, conditioner, guidance — всё подключается через slots
    - работает с ЛЮБОЙ модальностью (изображения, видео, аудио, 3D, молекулы, текст, временные ряды...)
    - можно прикручивать адаптеры, новые процессы, свои guidance
    - полностью совместима с training и deployment
    """
    
    block_type = "model/modular"
    block_version = "1.0.0"
    
    def __init__(self, config: DictConfig | dict):
        # Сначала инициализируем AbstractBlock (он создаст slots и подключит детей)
        AbstractBlock.__init__(self, config)
        # nn.Module уже вызван внутри AbstractBlock
        
        # Кэш для ускорения (особенно важно для видео/3D)
        self._cached_timestep_emb = None
        self.is_training = False
    
    def _define_slots(self) -> Dict[str, Slot]:
        """Определяем все Lego-дырки модели."""
        return {
            # Основные кирпичики
            "backbone": Slot(
                name="backbone",
                accepts=AbstractBackbone,
                multiple=False,
                optional=False
            ),
            "codec": Slot(
                name="codec",
                accepts=AbstractLatentCodec,
                multiple=False,
                optional=True,           # Можно работать без VAE (например, pixel-space)
                default={"type": "identity"}
            ),
            "conditioner": Slot(
                name="conditioner",
                accepts=AbstractConditioner,
                multiple=True,           # Можно несколько: текст + ControlNet + IP-Adapter
                optional=True
            ),
            "guidance": Slot(
                name="guidance",
                accepts=AbstractGuidance,
                multiple=True,           # CFG + PAG + FreeU одновременно
                optional=True,
                default={"type": "cfg"}
            ),
            "position": Slot(
                name="position",
                accepts=AbstractPositionEmbedder,
                multiple=False,
                optional=True,
                default={"type": "rope_nd"}
            ),
            
            # Адаптеры (LoRA, ControlNet, IP-Adapter и т.д.)
            "adapters": Slot(
                name="adapters",
                accepts=AbstractBlock,   # Любой адаптер из blocks/adapters/
                multiple=True,
                optional=True
            ),
            
            # Диффузионный процесс (опционально — можно вынести в sampler)
            "diffusion_process": Slot(
                name="diffusion_process",
                accepts=AbstractDiffusionProcess,
                multiple=False,
                optional=True
            )
        }
    
    def _forward_impl(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Dict[str, Any] | None = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        """Основной forward — работает с ЛЮБОЙ модальностью."""
        
        # 1. Кодирование в латент (если есть codec)
        latents = self._encode(x) if self.has_slot("codec") else x
        
        # 2. Позиционные эмбеддинги (работает в любой размерности)
        pos_emb = self.children["position"](t, latents.shape) if self.has_slot("position") else None
        
        # 3. Обработка условий (текст, ControlNet, IP-Adapter и т.д.)
        cond_emb = self._process_conditions(condition) if condition else None
        
        # 4. Прогон через backbone (UNet, DiT, Transformer — всё равно)
        backbone_output = self.children["backbone"](
            latents,
            timestep=t,
            condition=cond_emb,
            position_embedding=pos_emb
        )
        
        # 5. Применяем guidance (CFG, PAG, FreeU и кастомные)
        model_output = self._apply_guidance(backbone_output, condition)
        
        # 6. Декодирование обратно (если нужно)
        if self.has_slot("codec") and not self.is_training:
            model_output = self._decode_output(model_output)
        
        if return_dict:
            return {
                "noise_pred": model_output,
                "x0_pred": self._predict_x0(model_output, latents, t),
                "velocity": self._predict_velocity(model_output, latents, t),
                "latents": latents
            }
        return model_output
    
    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================
    
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Кодирование данных в латентное пространство."""
        if "codec" in self.children:
            return self.children["codec"].encode(x)
        return x
    
    def _decode_output(self, output: torch.Tensor) -> torch.Tensor:
        """Декодирование предсказания (обычно только для x0)."""
        if "codec" in self.children:
            return self.children["codec"].decode(output)
        return output
    
    def _process_conditions(self, condition: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Обработка всех conditioners (мультимодально)."""
        cond_emb = {}
        for conditioner in self.children.get("conditioner", []):
            emb = conditioner(condition)
            cond_emb.update(emb)  # можно несколько эмбеддингов
        return cond_emb
    
    def _apply_guidance(self, output: torch.Tensor, condition: Dict | None) -> torch.Tensor:
        """Применяем все guidance блоки по очереди."""
        result = output
        for guidance in self.children.get("guidance", []):
            result = guidance(result, condition=condition, model=self)
        return result
    
    def _predict_x0(self, noise_pred: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Математическая абстракция (работает для DDPM, Flow, EDM и т.д.)."""
        if "diffusion_process" in self.children:
            return self.children["diffusion_process"].predict_x0(noise_pred, x, t)
        # Fallback — простая формула DDPM
        alpha = self._get_alpha(t)
        return (x - (1 - alpha).sqrt() * noise_pred) / alpha.sqrt()
    
    def _predict_velocity(self, noise_pred: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Для flow-matching моделей."""
        if "diffusion_process" in self.children:
            return self.children["diffusion_process"].predict_velocity(noise_pred, x, t)
        return noise_pred  # по умолчанию
    
    def _get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Простой fallback для alpha (можно переопределить в diffusion_process)."""
        return torch.cos(t * 0.5 * torch.pi) ** 2
    
    # ==================== LEGO-МЕТОДЫ ====================
    
    def has_slot(self, slot_name: str) -> bool:
        """Проверка существования слота."""
        return slot_name in self.slots
    
    def attach_adapter(self, adapter: AbstractBlock):
        """Удобный метод для прикручивания LoRA / ControlNet / IP-Adapter."""
        self.attach_slot("adapters", adapter)
        # Автоматически применяем адаптер к backbone (стандартное поведение)
        if hasattr(adapter, "inject_into"):
            adapter.inject_into(self.children["backbone"])
    
    def set_training_mode(self, mode: bool = True):
        """Переключаем режим обучения/инференса."""
        self.is_training = mode
        self.train(mode)
    
    # ==================== ИНТЕРФЕЙС ДЛЯ ТРЕНИНГА И ИНФЕРЕНСА ====================
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Публичный метод для датасетов."""
        return self._encode(data)
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Публичный метод для визуализации."""
        return self.children.get("codec", lambda x: x).decode(latents)
    
    def forward_for_loss(self, x: torch.Tensor, t: torch.Tensor, condition: Dict | None = None) -> Dict[str, torch.Tensor]:
        """Специальный forward только для обучения (без лишних decode)."""
        self.set_training_mode(True)
        return self._forward_impl(x, t, condition, return_dict=True)
    
    def generate(self, condition: Dict[str, Any], **kwargs) -> torch.Tensor:
        """Быстрый shortcut для генерации (полный цикл через sampler)."""
        # Это заглушка — настоящий generate будет в engine/sampler
        # Но удобно иметь здесь для быстрого тестирования
        from ...core.engine.sampler import DiffusionSampler
        sampler = DiffusionSampler(self)
        return sampler.sample(condition=condition, **kwargs)
    
    def __repr__(self):
        slots = [f"{k}={len(v) if isinstance(v, list) else 1}" for k, v in self.children.items()]
        return f"<ModularDiffusionModel {self.block_id} | {' | '.join(slots)}>"