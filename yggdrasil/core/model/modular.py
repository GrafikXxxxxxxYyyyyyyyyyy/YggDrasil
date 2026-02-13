# yggdrasil/core/model/modular.py
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Any, Dict, Optional, Tuple, List

from yggdrasil.core.block.base import AbstractBlock
from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.slot import Slot


@register_block("model/modular")
class ModularDiffusionModel(AbstractBlock, nn.Module):
    """Единственная модель во всём YggDrasil.
    
    Это настоящий Lego-конструктор диффузии.
    Работает с любой модальностью и любыми адаптерами.
    """
    
    block_type = "model/modular"
    block_version = "1.0.0"
    
    def __init__(self, config: DictConfig | dict):
        # AbstractBlock.__init__ теперь сам вызывает nn.Module.__init__ в начале — _modules готов для add_module
        AbstractBlock.__init__(self, config)

        # Кэш
        self._cached_timestep_emb = None
        self.is_training = False
    
    def _define_slots(self) -> Dict[str, Slot]:
        """Определяем все Lego-слоты модели."""
        # Импортируем здесь, чтобы избежать circular import
        from .guidance import AbstractGuidance
        from .backbone import AbstractBackbone
        from .codec import AbstractLatentCodec
        from .conditioner import AbstractConditioner
        from .position import AbstractPositionEmbedder
        
        from yggdrasil.core.diffusion.process import AbstractDiffusionProcess
        
        return {
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
                optional=True,
                default={"type": "codec/identity"}
            ),
            "conditioner": Slot(
                name="conditioner",
                accepts=AbstractConditioner,
                multiple=True,
                optional=True
            ),
            "guidance": Slot(
                name="guidance",
                accepts=AbstractGuidance,
                multiple=True,
                optional=True,
                default={"type": "guidance/cfg"}
            ),
            "position": Slot(
                name="position",
                accepts=AbstractPositionEmbedder,
                multiple=False,
                optional=True,
                default={"type": "position/rope_nd"}
            ),
            "adapters": Slot(
                name="adapters",
                accepts=AbstractBlock,
                multiple=True,
                optional=True
            ),
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
        # Явно переносим на устройство и dtype модели (иначе часть вычислений может уйти на CPU)
        backbone_param = next(self._slot_children["backbone"].parameters(), None)
        if backbone_param is not None:
            device, dtype = backbone_param.device, backbone_param.dtype
            x = x.to(device=device, dtype=dtype)
            t = t.to(device=device)  # dtype таймстепа не приводим — UNet сам делает embedding
        else:
            x = x.to(dtype=torch.float32)

        # 1. Кодирование в латент только если вход в пиксельном пространстве (не латенты)
        if self.has_slot("codec"):
            codec = self._slot_children["codec"]
            latent_ch = getattr(codec, "latent_channels", 4)
            if x.shape[1] == latent_ch:
                latents = x  # уже латенты (сэмплинг)
            else:
                latents = self._encode(x)
        else:
            latents = x
        
        # 2. Позиционные эмбеддинги (если слот заполнен)
        pos_emb = None
        if self._slot_children.get("position") is not None:
            pos_emb = self._slot_children["position"](t, latents.shape)
        
        # 3. Обработка условий
        cond_emb = self._process_conditions(condition) if condition else None
        
        # 4. Прогон через backbone
        backbone_output = self._slot_children["backbone"](
            latents,
            timestep=t,
            condition=cond_emb,
            position_embedding=pos_emb
        )
        
        # 5. Применяем guidance (CFG и другие)
        model_output = self._apply_guidance(backbone_output, condition, latents, t)
        
        # Декодирование делается один раз в sampler после цикла, не по шагам
        if return_dict:
            return {
                "noise_pred": model_output,
                "x0_pred": self._predict_x0(model_output, latents, t),
                "velocity": self._predict_velocity(model_output, latents, t),
                "latents": latents
            }
        return model_output
    
    # ==================== GUIDANCE ====================
    
    def _apply_guidance(
        self,
        output: torch.Tensor,
        condition: Dict | None,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Применяем все guidance блоки."""
        result = output
        for guidance in self._slot_children.get("guidance", []):
            result = guidance(
                result,
                condition=condition,
                model=self,
                x=x,
                t=t
            )
        return result
    
    # ==================== ВСПОМОГАТЕЛЬНЫЕ ====================
    
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if "codec" in self._slot_children:
            return self._slot_children["codec"].encode(x)
        return x
    
    def _decode_output(self, output: torch.Tensor) -> torch.Tensor:
        if "codec" in self._slot_children:
            return self._slot_children["codec"].decode(output)
        return output
    
    def _process_conditions(self, condition: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        cond_emb = {}
        for conditioner in self._slot_children.get("conditioner", []):
            emb = conditioner(condition)
            cond_emb.update(emb)
        return cond_emb
    
    def _predict_x0(self, noise_pred: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if "diffusion_process" in self._slot_children:
            return self._slot_children["diffusion_process"].predict_x0(noise_pred, x, t)
        # Fallback DDPM
        alpha = torch.cos(t * 0.5 * torch.pi) ** 2
        return (x - (1 - alpha).sqrt() * noise_pred) / alpha.sqrt()
    
    def _predict_velocity(self, noise_pred: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if "diffusion_process" in self._slot_children:
            return self._slot_children["diffusion_process"].predict_velocity(noise_pred, x, t)
        return noise_pred
    
    # ==================== LEGO-МЕТОДЫ ====================
    
    def has_slot(self, slot_name: str) -> bool:
        return slot_name in self.slots
    
    def attach_adapter(self, adapter: AbstractBlock):
        self.attach_slot("adapters", adapter)
        if hasattr(adapter, "inject_into"):
            adapter.inject_into(self._slot_children["backbone"])
    
    def set_training_mode(self, mode: bool = True):
        self.is_training = mode
        self.train(mode)
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        return self._encode(data)
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self._slot_children.get("codec", lambda x: x).decode(latents)
    
    def forward_for_loss(self, x: torch.Tensor, t: torch.Tensor, condition: Dict | None = None) -> Dict[str, torch.Tensor]:
        self.set_training_mode(True)
        return self._forward_impl(x, t, condition, return_dict=True)
    
    def generate(self, condition: Dict[str, Any], **kwargs) -> torch.Tensor:
        from yggdrasil.core.engine.sampler import DiffusionSampler
        sampler = DiffusionSampler({"model": self, **kwargs})
        return sampler.sample(condition=condition, **kwargs)
    
    def __repr__(self):
        slots = [f"{k}={len(v) if isinstance(v, list) else 1}" for k, v in self._slot_children.items()]
        return f"<ModularDiffusionModel {self.block_id} | {' | '.join(slots)}>"