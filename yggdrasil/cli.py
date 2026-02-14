# yggdrasil/cli.py
"""CLI entry point для YggDrasil.

Использование:
    # Запустить Gradio UI
    python -m yggdrasil ui
    python -m yggdrasil ui --port 7860 --share
    
    # Запустить API сервер
    python -m yggdrasil api
    python -m yggdrasil api --port 8000 --host 0.0.0.0
    
    # Запустить оба (Gradio + API)
    python -m yggdrasil serve
    python -m yggdrasil serve --port 7860 --api-port 8000
    
    # Генерация из CLI
    python -m yggdrasil generate --preset sd15 --prompt "a cat" --output cat.png
    
    # Список пресетов
    python -m yggdrasil presets
    
    # Список блоков
    python -m yggdrasil blocks
    
    # Обучение
    python -m yggdrasil train --preset sd15 --data ./images --mode adapter --epochs 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="yggdrasil",
        description="YggDrasil — Universal Diffusion Framework",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # === UI ===
    ui_parser = subparsers.add_parser("ui", help="Launch Gradio UI")
    ui_parser.add_argument("--port", type=int, default=7860)
    ui_parser.add_argument("--host", type=str, default="0.0.0.0")
    ui_parser.add_argument("--share", action="store_true", help="Create public link")
    
    # === API ===
    api_parser = subparsers.add_parser("api", help="Launch FastAPI server")
    api_parser.add_argument("--port", type=int, default=8000)
    api_parser.add_argument("--host", type=str, default="0.0.0.0")
    api_parser.add_argument("--api-key", type=str, default=None)
    
    # === SERVE (UI + API) ===
    serve_parser = subparsers.add_parser("serve", help="Launch both Gradio UI and API")
    serve_parser.add_argument("--port", type=int, default=7860, help="Gradio port")
    serve_parser.add_argument("--api-port", type=int, default=8000, help="API port")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--share", action="store_true")
    serve_parser.add_argument("--preload", type=str, nargs="*", help="Preload model presets")
    
    # === GENERATE ===
    gen_parser = subparsers.add_parser("generate", help="Generate from CLI")
    gen_parser.add_argument("--preset", type=str, default="sd15", help="Model preset")
    gen_parser.add_argument("--prompt", type=str, required=True)
    gen_parser.add_argument("--negative", type=str, default="")
    gen_parser.add_argument("--output", type=str, default="output.png")
    gen_parser.add_argument("--steps", type=int, default=28)
    gen_parser.add_argument("--cfg", type=float, default=7.5)
    gen_parser.add_argument("--width", type=int, default=512)
    gen_parser.add_argument("--height", type=int, default=512)
    gen_parser.add_argument("--seed", type=int, default=42)
    
    # === PRESETS ===
    subparsers.add_parser("presets", help="List available presets")
    
    # === BLOCKS ===
    subparsers.add_parser("blocks", help="List registered blocks")
    
    # === TRAIN ===
    train_parser = subparsers.add_parser("train", help="Train model/adapter")
    train_parser.add_argument("--preset", type=str, required=True, help="Model preset")
    train_parser.add_argument("--data", type=str, required=True, help="Dataset path")
    train_parser.add_argument("--mode", type=str, default="adapter", choices=["full", "adapter", "finetune"])
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == "ui":
        _cmd_ui(args)
    elif args.command == "api":
        _cmd_api(args)
    elif args.command == "serve":
        _cmd_serve(args)
    elif args.command == "generate":
        _cmd_generate(args)
    elif args.command == "presets":
        _cmd_presets()
    elif args.command == "blocks":
        _cmd_blocks()
    elif args.command == "train":
        _cmd_train(args)


def _cmd_ui(args):
    """Запустить Gradio UI."""
    from .serving.gradio_ui import create_ui
    from .serving.schema import ServerConfig
    
    config = ServerConfig(host=args.host, port=args.port)
    demo = create_ui(config=config)
    print(f"Запуск YggDrasil UI на http://{args.host}:{args.port}")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


def _cmd_api(args):
    """Запустить FastAPI."""
    import uvicorn
    from .serving.api import create_api
    from .serving.schema import ServerConfig
    
    config = ServerConfig(host=args.host, api_port=args.port, api_key=args.api_key)
    app = create_api(config)
    print(f"Запуск YggDrasil API на http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def _cmd_serve(args):
    """Запустить Gradio + API одновременно."""
    import threading
    import uvicorn
    from .serving.api import create_api
    from .serving.gradio_ui import create_ui
    from .serving.schema import ServerConfig
    
    # Preload models
    preload = []
    if args.preload:
        for preset_name in args.preload:
            preload.append({"model_id": preset_name, "pretrained": preset_name})
    
    config = ServerConfig(
        host=args.host,
        port=args.port,
        api_port=args.api_port,
        preload_models=preload,
    )
    
    # Создаём API
    api_app = create_api(config)
    manager = api_app.state.manager
    
    # API в отдельном потоке
    api_thread = threading.Thread(
        target=lambda: uvicorn.run(api_app, host=args.host, port=args.api_port),
        daemon=True,
    )
    api_thread.start()
    print(f"API запущен на http://{args.host}:{args.api_port}")
    
    # Gradio в основном потоке (с тем же manager)
    demo = create_ui(manager=manager, config=config)
    print(f"UI запущен на http://{args.host}:{args.port}")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


def _cmd_generate(args):
    """Генерация из CLI."""
    import torch
    from PIL import Image
    
    from .configs import get_preset
    from .core.block.registry import auto_discover
    from .integration.diffusers import load_from_diffusers
    from .core.engine.sampler import DiffusionSampler
    
    auto_discover()
    
    print(f"Загрузка пресета: {args.preset}")
    preset = get_preset(args.preset)
    
    pretrained = preset.model.backbone.get("pretrained", "runwayml/stable-diffusion-v1-5")
    
    model = load_from_diffusers(pretrained)
    
    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    model = model.to(device)
    
    sampler_cfg = {
        "num_inference_steps": args.steps,
        "guidance_scale": args.cfg,
    }
    if "diffusion_process" in preset:
        sampler_cfg["diffusion_process"] = dict(preset.diffusion_process)
    if "solver" in preset:
        sampler_cfg["solver"] = dict(preset.solver)
    
    sampler = DiffusionSampler(sampler_cfg, model=model).to(device)
    
    print(f"Генерация: {args.prompt}")
    
    condition = {"text": args.prompt}
    if args.negative:
        condition["negative_text"] = args.negative
    
    h = args.height // 8
    w = args.width // 8
    
    gen_device = device if device != "mps" else "cpu"
    generator = torch.Generator(gen_device).manual_seed(args.seed)
    
    result = sampler.sample(
        condition=condition,
        shape=(1, 4, h, w),
        generator=generator if device != "mps" else None,
    )
    
    # Save
    img = (result / 2 + 0.5).clamp(0, 1)
    img = (img * 255).to(torch.uint8).cpu().numpy()[0].transpose(1, 2, 0)
    pil = Image.fromarray(img)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(output_path)
    print(f"Сохранено: {output_path}")


def _cmd_presets():
    """Список пресетов."""
    from .configs import list_presets, get_preset
    
    presets = list_presets()
    if not presets:
        print("Нет доступных пресетов")
        return
    
    print("Доступные пресеты:\n")
    for name in presets:
        try:
            cfg = get_preset(name)
            meta = cfg.get("_meta", {})
            desc = meta.get("description", "")
            modality = meta.get("modality", "unknown")
            print(f"  {name:20s} | {modality:12s} | {desc}")
        except Exception:
            print(f"  {name:20s} | (error loading)")
    print()


def _cmd_blocks():
    """Список блоков."""
    from .core.block.registry import auto_discover, list_blocks
    
    auto_discover()
    blocks = list_blocks()
    
    if not blocks:
        print("Нет зарегистрированных блоков")
        return
    
    categories = {}
    for key, cls in sorted(blocks.items()):
        cat = key.split("/")[0] if "/" in key else "other"
        categories.setdefault(cat, []).append((key, cls))
    
    print("Зарегистрированные блоки:\n")
    for cat, items in sorted(categories.items()):
        print(f"  [{cat.upper()}]")
        for key, cls in items:
            doc = (cls.__doc__ or "").split("\n")[0].strip()[:60]
            print(f"    {key:40s} | {doc}")
        print()


def _cmd_train(args):
    """Обучение из CLI."""
    import torch
    from .configs import get_preset
    from .core.block.registry import auto_discover
    from .integration.diffusers import load_from_diffusers
    from .training.trainer import DiffusionTrainer, TrainingConfig
    from .training.data import ImageFolderSource
    from .training.loss import EpsilonLoss, FlowMatchingLoss
    from .core.diffusion.ddpm import DDPMProcess
    from .core.diffusion.flow import RectifiedFlowProcess
    
    auto_discover()
    
    print(f"Загрузка пресета: {args.preset}")
    preset = get_preset(args.preset)
    
    pretrained = preset.model.backbone.get("pretrained", "runwayml/stable-diffusion-v1-5")
    model = load_from_diffusers(pretrained)
    
    # Process
    process_type = preset.get("diffusion_process", {}).get("type", "diffusion/process/ddpm")
    if "flow" in process_type:
        process = RectifiedFlowProcess({})
        loss_fn = FlowMatchingLoss()
    else:
        process = DDPMProcess(dict(preset.get("diffusion_process", {})))
        loss_fn = EpsilonLoss()
    
    config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        mixed_precision=args.precision,
        train_mode=args.mode,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    dataset = ImageFolderSource(args.data)
    trainer = DiffusionTrainer(model, process, loss_fn, config)
    
    print(f"Начинаем обучение: {args.mode} mode, {args.epochs} epochs")
    history = trainer.train(dataset)
    print("Обучение завершено!")


if __name__ == "__main__":
    main()
