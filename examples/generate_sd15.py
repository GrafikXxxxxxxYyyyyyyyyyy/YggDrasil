#!/usr/bin/env python3
"""YggDrasil — Stable Diffusion 1.5: полный пример генерации.

Демонстрирует три способа запуска SD 1.5:
1. Graph API (новый) — через ComputeGraph + шаблон  ← рекомендуемый
2. DiffusersBridge — импорт из HuggingFace Diffusers
3. Legacy Sampler — через DiffusionSampler + ModularDiffusionModel
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image

# ==================== НАСТРОЙКИ ====================

PROMPT = "a majestic cyberpunk samurai standing on a rainy neon rooftop at night, cinematic lighting, ultra detailed, 8k"
SEED = 42
STEPS = 28
CFG_SCALE = 7.5
WIDTH, HEIGHT = 512, 512


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_image(tensor: torch.Tensor, path: str):
    """Сохранить тензор [-1, 1] или [0, 1] как PNG."""
    img = tensor[0] if tensor.dim() == 4 else tensor
    img = img.cpu().float()
    if img.min() < 0:
        img = (img / 2 + 0.5)
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype("uint8")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    print(f"  Сохранено → {path}")


# ==================================================================
# СПОСОБ 1: Graph API (рекомендуемый)
# ==================================================================

def generate_graph_api():
    """Генерация через ComputeGraph + шаблон sd15_txt2img.

    Шаблон создаёт полный граф:
        conditioner_0 → [LoopSubGraph(backbone→guidance→solver)] → codec

    Все компоненты — отдельные Lego-блоки, которые можно заменить.
    """
    print("\n" + "=" * 60)
    print("СПОСОБ 1: Graph API (ComputeGraph + шаблон)")
    print("=" * 60)

    from yggdrasil.core.graph.graph import ComputeGraph

    device = get_device()
    steps = 5 if device in ("mps", "cpu") else STEPS

    # ── Всё в 3 строки ──
    graph = ComputeGraph.from_template("sd15_txt2img", device=device)
    print(f"  {graph}")
    print(f"  Узлы: {list(graph.nodes.keys())}")

    print(f"  Генерируем: \"{PROMPT}\" ({steps} шагов)")
    outputs = graph.execute(
        prompt=PROMPT,
        guidance_scale=CFG_SCALE,
        num_steps=steps,
        seed=SEED,
        width=WIDTH,
        height=HEIGHT,
    )

    save_image(outputs["decoded"], f"output/sd15_graph_{SEED}.png")
    return outputs


# ==================================================================
# СПОСОБ 2: DiffusersBridge (импорт из HuggingFace)
# ==================================================================

def generate_diffusers_bridge():
    """Генерация через DiffusersBridge — импорт модели из diffusers.

    DiffusersBridge автоматически:
    1. Загружает pipeline из HuggingFace
    2. Разбирает на компоненты (UNet, VAE, CLIP, Scheduler)
    3. Маппит каждый на YggDrasil-блок
    4. Собирает ComputeGraph
    """
    print("\n" + "=" * 60)
    print("СПОСОБ 2: DiffusersBridge (из HuggingFace Diffusers)")
    print("=" * 60)

    from yggdrasil.integration.diffusers import DiffusersBridge

    device = get_device()

    # 1. Загружаем и конвертируем
    print("  Загружаем из HuggingFace...")
    graph = DiffusersBridge.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    )
    print(f"  {graph}")
    print(f"  Узлы: {list(graph.nodes.keys())}")

    # 2. На устройство
    graph.to(device)

    # 3. Генерируем (DiffusersBridge создаёт single-step graph, нужен LoopSubGraph)
    from yggdrasil.core.graph.subgraph import LoopSubGraph

    step_graph = graph  # граф одного шага
    loop = LoopSubGraph.create(inner_graph=step_graph, num_iterations=STEPS)

    noise = torch.randn(1, 4, HEIGHT // 8, WIDTH // 8, device=device)
    print(f"  Генерируем: \"{PROMPT}\"")

    result = loop.process(
        initial_latents=noise,
        condition={"text": PROMPT},
    )

    print(f"  Bridge graph выполнен, ключи: {list(result.keys())}")
    return result


# ==================================================================
# СПОСОБ 3: Legacy Sampler (slot-based)
# ==================================================================

def generate_legacy_sampler():
    """Генерация через Legacy DiffusionSampler + ModularDiffusionModel.

    Это оригинальный API до перехода на графы.
    Работает через slot-based архитектуру.
    """
    print("\n" + "=" * 60)
    print("СПОСОБ 3: Legacy Sampler (slot-based)")
    print("=" * 60)

    from yggdrasil.integration.diffusers import load_stable_diffusion_15
    from yggdrasil.core.engine.sampler import DiffusionSampler

    device = get_device()

    # 1. Загружаем SD 1.5 как ModularDiffusionModel
    print("  Загружаем SD 1.5...")
    model = load_stable_diffusion_15(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    )
    model = model.to(device)
    print(f"  Модель на: {next(model.parameters(), torch.tensor(0)).device}")

    # 2. Создаём sampler
    sampler = DiffusionSampler(
        {
            "num_inference_steps": STEPS,
            "guidance_scale": CFG_SCALE,
            "diffusion_process": {"type": "diffusion/process/ddpm"},
            "solver": {"type": "diffusion/solver/ddim", "eta": 0.0},
        },
        model=model,
    )
    sampler = sampler.to(device)

    # 3. Генерируем
    print(f"  Генерируем: \"{PROMPT}\"")
    if device == "mps":
        g = torch.Generator().manual_seed(SEED)
    else:
        g = torch.Generator(device).manual_seed(SEED)

    image_tensor = sampler.sample(
        condition={"text": PROMPT},
        shape=(1, 4, HEIGHT // 8, WIDTH // 8),
        generator=g,
    )

    # 4. Сохраняем
    save_image(image_tensor, f"output/sd15_legacy_{SEED}.png")
    return image_tensor


# ==================================================================
# БОНУС: Lego-замена блоков
# ==================================================================

def demo_lego_replace():
    """Демонстрация замены блока в графе — суть Lego-конструктора.

    Меняем solver с DDIM на Euler, меняем guidance scale — и генерируем снова.
    """
    print("\n" + "=" * 60)
    print("БОНУС: Lego-замена блоков")
    print("=" * 60)

    from yggdrasil.core.graph.graph import ComputeGraph
    from yggdrasil.core.block.builder import BlockBuilder

    device = get_device()

    # 1. Базовый граф (device сразу указан)
    graph = ComputeGraph.from_template("sd15_txt2img", device=device)
    print(f"  Исходный граф: {graph}")

    # 2. Показываем как заменить блок
    loop_node = graph.nodes.get("denoise_loop")
    if loop_node and hasattr(loop_node, "graph"):
        inner = loop_node.graph
        print(f"  Внутренний граф цикла: {list(inner.nodes.keys())}")

        # Заменяем DDIM solver на Euler
        print("  Заменяем DDIM → Euler...")
        euler = BlockBuilder.build({"type": "solver/euler"})
        inner.replace_node("solver", euler)
        print(f"  Solver заменён на: {type(inner.nodes['solver']).__name__}")

    # 3. Показываем структуру графа (Mermaid)
    print("\n  Mermaid-диаграмма графа:")
    print(graph.visualize())


# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YggDrasil SD 1.5 примеры")
    parser.add_argument(
        "--mode",
        choices=["graph", "bridge", "legacy", "lego", "all"],
        default="graph",
        help="Какой способ запуска (default: graph)",
    )
    args = parser.parse_args()

    if args.mode == "graph" or args.mode == "all":
        generate_graph_api()

    if args.mode == "bridge" or args.mode == "all":
        generate_diffusers_bridge()

    if args.mode == "legacy" or args.mode == "all":
        generate_legacy_sampler()

    if args.mode == "lego" or args.mode == "all":
        demo_lego_replace()

    print("\nГотово!")
