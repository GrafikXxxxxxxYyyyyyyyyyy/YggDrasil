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
    """Загружает SD 1.5 как Lego-модель из конфига (все блоки YggDrasil)."""
    # Гарантированно регистрируем блоки до сборки
    import yggdrasil.blocks.backbones.unet_2d_condition  # noqa: F401
    import yggdrasil.blocks.codecs.autoencoder_kl  # noqa: F401
    import yggdrasil.blocks.conditioners.clip_text  # noqa: F401
    import yggdrasil.blocks.guidances.cfg  # noqa: F401

    fp16 = torch_dtype == torch.float16
    config = {
        "type": "model/modular",
        "id": "sd15_lego",
        "backbone": {
            "type": "backbone/unet2d_condition",
            "pretrained": pretrained_model_name_or_path,
            "fp16": fp16,
        },
        "codec": {
            "type": "codec/autoencoder_kl",
            "pretrained": pretrained_model_name_or_path,
            "fp16": fp16,
            "scaling_factor": 0.18215,
        },
        "conditioner": {
            "type": "conditioner/clip_text",
            "pretrained": pretrained_model_name_or_path,
            "tokenizer_subfolder": "tokenizer",
            "text_encoder_subfolder": "text_encoder",
            "max_length": 77,
        },
        "guidance": {"type": "guidance/cfg", "scale": 7.5},
    }
    print(f"Собираем SD 1.5 из {pretrained_model_name_or_path}...")
    model = BlockBuilder.build(config)
    print("Модель успешно собрана как Lego!")
    return model


# Удобные шорткаты
def load_stable_diffusion_15(pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5", **kwargs):
    return load_from_diffusers(pretrained_model_name_or_path, **kwargs)

def load_sdxl(**kwargs):
    return load_from_diffusers("stabilityai/stable-diffusion-xl-base-1.0", **kwargs)