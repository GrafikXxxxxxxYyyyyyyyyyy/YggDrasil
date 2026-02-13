import torch
from ..integration.diffusers import load_stable_diffusion_15
from ..core.engine.sampler import DiffusionSampler

# 1. Загружаем как Lego-модель
model = load_stable_diffusion_15(
    "runwayml/stable-diffusion-v1-5",
    fp16=True
).to("cuda")

# 2. Создаём сэмплер
sampler = DiffusionSampler({
    "type": "engine/sampler",
    "model": model,
    "num_inference_steps": 30,
    "guidance_scale": 7.5
})

# 3. Генерируем!
image = sampler.sample(
    condition={"text": "a beautiful cyberpunk city at night, highly detailed, 8k"},
    shape=(1, 4, 64, 64)   # latent shape для SD 1.5
)

# 4. Сохраняем
image = (image / 2 + 0.5).clamp(0, 1)
image = (image * 255).to(torch.uint8).cpu().numpy()[0].transpose(1, 2, 0)
from PIL import Image
Image.fromarray(image).save("sd15_lego_output.png")
print("Готово! Stable Diffusion 1.5 работает внутри YggDrasil как Lego.")