import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Явная регистрация ВСЕХ блоков (самый надёжный способ)
import yggdrasil.core.block.base
import yggdrasil.core.block.slot
import yggdrasil.core.block.registry
import yggdrasil.core.block.builder

import yggdrasil.core.model.modular
import yggdrasil.core.model.backbone
import yggdrasil.core.model.codec
import yggdrasil.core.model.conditioner
import yggdrasil.core.model.guidance
import yggdrasil.core.model.position

import yggdrasil.blocks.adapters.lora
import yggdrasil.blocks.adapters.controlnet
import yggdrasil.blocks.adapters.t2i_adapter
import yggdrasil.blocks.adapters.fusion

import torch
from PIL import Image
from pathlib import Path

from yggdrasil.integration.diffusers import load_stable_diffusion_15
from yggdrasil.core.engine.sampler import DiffusionSampler

# ==================== НАСТРОЙКИ ====================

PROMPT = "a majestic cyberpunk samurai standing on a rainy neon rooftop at night, cinematic lighting, ultra detailed, 8k"
NEGATIVE = "blurry, low quality, deformed, ugly, text"

SEED = 42
STEPS = 28
CFG = 7.5

# ==================== ЗАПУСК ====================

print("Загружаем SD 1.5 как Lego-модель...")
model = load_stable_diffusion_15(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)

sampler = DiffusionSampler({
    "model": model,
    "num_inference_steps": STEPS,
    "guidance_scale": CFG,
})

print(f"Генерируем: {PROMPT}")

condition = {"text": PROMPT}

image_tensor = sampler.sample(
    condition=condition,
    shape=(1, 4, 64, 64),
    generator=torch.Generator("cuda").manual_seed(SEED)
)

# Декодируем
image = model.children["codec"].decode(image_tensor)
image = (image / 2 + 0.5).clamp(0, 1)
image = (image * 255).to(torch.uint8).cpu().numpy()[0].transpose(1, 2, 0)

img = Image.fromarray(image)
output_path = Path("output") / f"sd15_{SEED}.png"
output_path.parent.mkdir(exist_ok=True)
img.save(output_path)

print(f"Готово → {output_path}")
img.show()