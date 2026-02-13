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

import yggdrasil.core.diffusion.ddpm
import yggdrasil.core.diffusion.solver.ddim

# Конкретные блоки для SD 1.5 (обязательно до load_stable_diffusion_15)
import yggdrasil.blocks.backbones.unet_2d_condition
import yggdrasil.blocks.codecs.autoencoder_kl
import yggdrasil.blocks.conditioners.clip_text
import yggdrasil.blocks.guidances.cfg

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

def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE_STR = _get_device()
DEVICE = torch.device(DEVICE_STR)
print(f"Устройство: {DEVICE}")

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

model = model.to(DEVICE)
print(f"Модель после .to(DEVICE): {next(model.parameters(), torch.tensor(0)).device}")

sampler = DiffusionSampler(
    {
        "num_inference_steps": STEPS,
        "guidance_scale": CFG,
        "diffusion_process": {"type": "diffusion/process/ddpm"},
        "solver": {"type": "diffusion/solver/ddim", "eta": 0.0},
    },
    model=model,
)
# Явно переносим сэмплер и модель на устройство (модель могла не попасть в module tree сэмплера до add_module)
sampler = sampler.to(DEVICE)
sampler._slot_children["model"].to(DEVICE)

print(f"Модель на устройстве: {next(sampler._slot_children['model'].parameters(), torch.tensor(0)).device}")

print(f"Генерируем: {PROMPT}")

condition = {"text": PROMPT}

# sample() уже возвращает декодированное изображение из VAE
image_tensor = sampler.sample(
    condition=condition,
    shape=(1, 4, 64, 64),
    generator=torch.Generator(DEVICE_STR).manual_seed(SEED),
)

# Нормализация из [-1, 1] в [0, 255] и в формат (H, W, C)
image = (image_tensor / 2 + 0.5).clamp(0, 1)
image = (image * 255).to(torch.uint8).cpu().numpy()[0].transpose(1, 2, 0)

img = Image.fromarray(image)
output_path = Path("output") / f"sd15_{SEED}.png"
output_path.parent.mkdir(exist_ok=True)
img.save(output_path)

print(f"Готово → {output_path}")
img.show()