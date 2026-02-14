# yggdrasil/__init__.py
"""YggDrasil — Universal Lego Diffusion Framework.

Build your diffusion like Lego, even if it hasn't been invented yet.

Использование:
    # Из Python
    from yggdrasil import load_model, generate, list_presets
    model = load_model("sd15")
    image = generate(model, prompt="a cat")
    
    # Из CLI
    python -m yggdrasil ui              # Gradio UI
    python -m yggdrasil api             # FastAPI endpoint
    python -m yggdrasil serve           # UI + API
    python -m yggdrasil generate ...    # CLI генерация
    python -m yggdrasil train ...       # Обучение
"""
__version__ = "0.1.0"

# Workaround: TensorFlow Metal конфликтует с PyTorch MPS при параллельной загрузке.
# Предотвращаем краш "platform is already registered with name: METAL",
# патчим __spec__ на уже-загруженных стабах tensorflow, чтобы diffusers мог импортироваться.
def _patch_tensorflow_compat():
    import sys
    tf = sys.modules.get("tensorflow")
    if tf is not None and getattr(tf, "__spec__", None) is None:
        import importlib.machinery
        tf.__spec__ = importlib.machinery.ModuleSpec("tensorflow", None)
        if not hasattr(tf, "__path__"):
            tf.__path__ = []

_patch_tensorflow_compat()

# Авто-регистрация блоков
from yggdrasil.core.block.registry import auto_discover as _auto_discover
_auto_discover()

# === Удобные функции верхнего уровня ===

def load_model(preset_or_path: str, **kwargs):
    """Загрузить модель из пресета или pretrained пути.
    
    Args:
        preset_or_path: Имя пресета ("sd15", "sdxl", "flux_dev") 
                       или HuggingFace ID ("runwayml/stable-diffusion-v1-5")
                       или путь к YAML конфигу
    
    Returns:
        ModularDiffusionModel
    """
    from pathlib import Path
    
    # 1. Проверяем — это пресет?
    from .configs import list_presets, get_preset
    if preset_or_path in list_presets():
        config = get_preset(preset_or_path)
        pretrained = config.model.backbone.get("pretrained", None)
        if pretrained:
            from .integration.diffusers import load_from_diffusers
            return load_from_diffusers(pretrained, **kwargs)
    
    # 2. Проверяем — это YAML файл?
    if Path(preset_or_path).suffix in (".yaml", ".yml") and Path(preset_or_path).exists():
        from .core.block.builder import BlockBuilder
        return BlockBuilder.build(preset_or_path)
    
    # 3. Считаем это HuggingFace ID
    from .integration.diffusers import load_from_diffusers
    return load_from_diffusers(preset_or_path, **kwargs)


def generate(model, prompt: str, **kwargs):
    """Быстрая генерация.
    
    Args:
        model: ModularDiffusionModel
        prompt: Текстовое условие
        **kwargs: steps, cfg, width, height, seed, ...
    
    Returns:
        PIL.Image для изображений, torch.Tensor для других модальностей
    """
    import torch
    from .core.engine.sampler import DiffusionSampler
    
    steps = kwargs.pop("steps", 28)
    cfg = kwargs.pop("cfg", 7.5)
    width = kwargs.pop("width", 512)
    height = kwargs.pop("height", 512)
    seed = kwargs.pop("seed", 42)
    
    device = next(model.parameters()).device
    device_str = str(device)
    
    sampler_cfg = {
        "num_inference_steps": steps,
        "guidance_scale": cfg,
    }
    
    # Подключаем диффузионный процесс если передан
    if "diffusion_process" in kwargs:
        sampler_cfg["diffusion_process"] = kwargs.pop("diffusion_process")
    else:
        sampler_cfg["diffusion_process"] = {"type": "diffusion/process/ddpm"}
    
    if "solver" in kwargs:
        sampler_cfg["solver"] = kwargs.pop("solver")
    else:
        sampler_cfg["solver"] = {"type": "diffusion/solver/ddim", "eta": 0.0}
    
    sampler = DiffusionSampler(sampler_cfg, model=model).to(device)
    
    condition = {"text": prompt}
    h, w = height // 8, width // 8
    
    if "mps" in device_str:
        generator = None
    else:
        generator = torch.Generator(device_str).manual_seed(seed)
    
    result = sampler.sample(
        condition=condition,
        shape=(1, 4, h, w),
        generator=generator,
    )
    
    # Конвертируем в PIL Image
    try:
        from PIL import Image
        img = (result / 2 + 0.5).clamp(0, 1)
        img = (img * 255).to(torch.uint8).cpu().numpy()[0].transpose(1, 2, 0)
        return Image.fromarray(img)
    except Exception:
        return result


def list_presets():
    """Список доступных пресетов."""
    from .configs import list_presets as _list
    return _list()


def list_blocks():
    """Список зарегистрированных блоков."""
    from .core.block.registry import list_blocks as _list
    return _list()
