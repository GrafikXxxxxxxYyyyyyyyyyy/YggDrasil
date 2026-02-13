# yggdrasil/integration/diffusers.py
from __future__ import annotations

import torch
from typing import Optional, Dict, Any, List

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    FluxPipeline,
)

from yggdrasil.core.block.builder import BlockBuilder
from yggdrasil.core.model.modular import ModularDiffusionModel

# Явно импортируем все нужные блоки, чтобы они зарегистрировались
import yggdrasil.core.model.modular
import yggdrasil.blocks.backbones.unet_2d_condition
import yggdrasil.blocks.codecs.autoencoder_kl
import yggdrasil.blocks.conditioners.clip_text
import yggdrasil.blocks.guidances.cfg


def load_from_diffusers(
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
    torch_dtype: torch.dtype = torch.float16,
    **kwargs
) -> ModularDiffusionModel:
    """Загружает SD 1.5 / SDXL / Flux как полноценную Lego-модель."""
    
    print(f"Загружаем {pretrained_model_name_or_path}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        safety_checker=None,
        **kwargs
    )
    
    # Создаём базовую модель
    model = ModularDiffusionModel({
        "type": "model/modular",
        "guidance": {"type": "guidance/cfg", "scale": 7.5},
    })
    
    # Прикрепляем готовые компоненты из пайплайна
    model.attach_slot("backbone", pipe.unet)
    model.attach_slot("codec", pipe.vae)
    
    # Conditioner (CLIP)
    from yggdrasil.blocks.conditioners.clip_text import CLIPTextConditioner
    conditioner = CLIPTextConditioner({})
    model.attach_slot("conditioner", conditioner)
    
    print("Модель успешно собрана как Lego!")
    return model


# Удобные шорткаты
def load_stable_diffusion_15(**kwargs):
    return load_from_diffusers("runwayml/stable-diffusion-v1-5", **kwargs)

def load_sdxl(**kwargs):
    return load_from_diffusers("stabilityai/stable-diffusion-xl-base-1.0", **kwargs)