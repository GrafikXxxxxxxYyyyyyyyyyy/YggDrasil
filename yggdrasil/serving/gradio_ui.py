# yggdrasil/serving/gradio_ui.py
"""Gradio UI для YggDrasil — универсальный интерфейс для любой диффузионной модели.

Три вкладки:
    1. Generate — генерация (любая модальность)
    2. Models   — управление моделями (загрузка/выгрузка/конструктор)
    3. Train    — обучение модели / адаптера

Архитектура:
    UI ↔ ModelManager (shared with API)
    Один ModelManager обслуживает и Gradio, и REST API одновременно.
"""
from __future__ import annotations

import io
import base64
import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image

from .schema import GenerateRequest, OutputFormat, ServerConfig


# ==================== HELPER ====================

def _get_device_info() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
        return f"CUDA: {name} ({mem:.1f} GB)"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "Apple MPS"
    return "CPU"


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Тензор [-1, 1] → PIL Image."""
    img = (tensor / 2 + 0.5).clamp(0, 1)
    img = (img * 255).to(torch.uint8).cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    if img.shape[0] in (1, 3, 4):
        img = img.transpose(1, 2, 0)
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
    return Image.fromarray(img)


# ==================== MAIN UI ====================

def create_ui(
    manager: Optional["ModelManager"] = None,
    config: Optional[ServerConfig] = None,
    share: bool = False,
) -> "gr.Blocks":
    """Создать Gradio интерфейс.
    
    Args:
        manager: ModelManager (общий с API). Если None — создаём новый.
        config: Конфигурация сервера.
        share: Создать публичную ссылку (для удалённого доступа).
    """
    import gradio as gr
    
    if manager is None:
        from .api import ModelManager
        manager = ModelManager(config or ServerConfig())
    
    # ==================== GENERATION TAB ====================
    
    def generate_image(
        model_id: str,
        prompt: str,
        negative_prompt: str,
        steps: int,
        cfg_scale: float,
        width: int,
        height: int,
        seed: int,
        batch_size: int,
    ) -> Tuple[List[Image.Image], str]:
        """Генерация изображений."""
        if not model_id or model_id not in manager.samplers:
            return [], f"Модель '{model_id}' не загружена. Загрузите модель во вкладке Models."
        
        sampler = manager.samplers[model_id]
        model = manager.models[model_id]
        device = next(model.parameters()).device
        device_str = str(device)
        
        actual_seed = seed if seed >= 0 else int(torch.randint(0, 2**32, (1,)).item())
        
        # Generator (MPS не поддерживает)
        if "mps" in device_str:
            generator = None
        else:
            generator = torch.Generator(device_str).manual_seed(actual_seed)
        
        condition = {"text": prompt}
        if negative_prompt:
            condition["negative_text"] = negative_prompt
        
        h_latent = height // 8
        w_latent = width // 8
        shape = (batch_size, 4, h_latent, w_latent)
        
        start_time = time.time()
        
        try:
            with torch.no_grad():
                result = sampler.sample(
                    condition=condition,
                    shape=shape,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    generator=generator,
                )
            
            elapsed = time.time() - start_time
            
            # Конвертируем в PIL
            images = []
            if result.ndim == 4:
                for i in range(result.shape[0]):
                    images.append(_tensor_to_pil(result[i:i+1]))
            else:
                images.append(_tensor_to_pil(result))
            
            info = (
                f"Seed: {actual_seed} | Steps: {steps} | CFG: {cfg_scale} | "
                f"Size: {width}x{height} | Time: {elapsed:.1f}s | Device: {device_str}"
            )
            return images, info
            
        except Exception as e:
            return [], f"Ошибка генерации: {e}"
    
    # ==================== MODELS TAB ====================
    
    def load_model_ui(
        model_id: str,
        pretrained_path: str,
        model_config_json: str,
    ) -> Tuple[str, str]:
        """Загрузка модели через UI."""
        import asyncio
        
        if not model_id:
            return "Введите ID модели", get_model_list()
        
        try:
            config_dict = json.loads(model_config_json) if model_config_json.strip() else None
            pretrained = pretrained_path.strip() if pretrained_path.strip() else None
            
            if not pretrained and not config_dict:
                return "Укажите pretrained path или JSON конфиг модели", get_model_list()
            
            loop = asyncio.new_event_loop()
            info = loop.run_until_complete(
                manager.load_model(model_id, config=config_dict, pretrained=pretrained)
            )
            loop.close()
            
            return f"Модель {model_id} загружена! ({info.num_parameters:,} параметров, {info.device})", get_model_list()
        except Exception as e:
            return f"Ошибка: {e}", get_model_list()
    
    def unload_model_ui(model_id: str) -> Tuple[str, str]:
        """Выгрузка модели."""
        import asyncio
        if model_id and model_id in manager.models:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(manager.unload_model(model_id))
            loop.close()
            return f"Модель {model_id} выгружена", get_model_list()
        return "Модель не найдена", get_model_list()
    
    def get_model_list() -> str:
        """Получить список моделей в текстовом формате."""
        if not manager.model_info:
            return "Нет загруженных моделей"
        
        lines = []
        for mid, info in manager.model_info.items():
            status_icon = {"ready": "🟢", "loading": "🟡", "error": "🔴", "unloaded": "⚪"}.get(info.status.value, "❓")
            lines.append(
                f"{status_icon} {mid} | {info.status.value} | "
                f"{info.num_parameters:,} params | {info.device}"
            )
        return "\n".join(lines)
    
    def get_model_choices() -> List[str]:
        """Список ID загруженных моделей для dropdown."""
        return [mid for mid, info in manager.model_info.items() if info.status.value == "ready"]
    
    # ==================== TRAINING TAB ====================
    
    def start_training_ui(
        model_id: str,
        dataset_path: str,
        train_mode: str,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        mixed_precision: str,
        use_ema: bool,
        save_every: int,
    ) -> str:
        """Запуск обучения через UI."""
        if model_id not in manager.models:
            return f"Модель {model_id} не загружена"
        
        if not dataset_path or not Path(dataset_path).exists():
            return f"Путь к данным не существует: {dataset_path}"
        
        try:
            from ..training.trainer import DiffusionTrainer, TrainingConfig
            from ..training.data import ImageFolderSource
            from ..core.diffusion.ddpm import DDPMProcess
            from ..training.loss import EpsilonLoss
            
            config = TrainingConfig(
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                mixed_precision=mixed_precision,
                use_ema=use_ema,
                save_every=save_every,
                train_mode=train_mode,
            )
            
            model = manager.models[model_id]
            trainer = DiffusionTrainer(model, DDPMProcess(), EpsilonLoss(), config)
            dataset = ImageFolderSource(dataset_path)
            
            # Запускаем в отдельном потоке
            import threading
            def run():
                try:
                    trainer.train(dataset)
                except Exception as e:
                    print(f"Training error: {e}")
            
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            
            return f"Обучение запущено для {model_id} | Mode: {train_mode} | Epochs: {num_epochs} | BS: {batch_size}"
        except Exception as e:
            return f"Ошибка: {e}"
    
    # ==================== BUILD UI ====================
    
    with gr.Blocks(
        title="YggDrasil — Universal Diffusion Framework",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
        ),
        css="""
        .main-title {
            text-align: center;
            margin-bottom: 0.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 1.5em;
        }
        """,
    ) as demo:
        
        gr.HTML("""
        <h1 class="main-title">🌳 YggDrasil</h1>
        <p class="subtitle">Universal Diffusion Framework — Build your diffusion like Lego</p>
        """)
        
        with gr.Tabs():
            
            # =============== TAB 1: GENERATE ===============
            with gr.Tab("Generate", id="generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_dropdown = gr.Dropdown(
                            label="Model",
                            choices=get_model_choices(),
                            value=None,
                            interactive=True,
                        )
                        refresh_models_btn = gr.Button("🔄 Refresh Models", size="sm")
                        
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="a majestic cyberpunk samurai standing on a rainy neon rooftop...",
                            lines=3,
                        )
                        negative_input = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="blurry, low quality, deformed",
                            lines=2,
                        )
                        
                        with gr.Row():
                            steps_slider = gr.Slider(1, 150, value=28, step=1, label="Steps")
                            cfg_slider = gr.Slider(0, 30, value=7.5, step=0.5, label="CFG Scale")
                        
                        with gr.Row():
                            width_slider = gr.Slider(128, 2048, value=512, step=64, label="Width")
                            height_slider = gr.Slider(128, 2048, value=512, step=64, label="Height")
                        
                        with gr.Row():
                            seed_input = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                            batch_slider = gr.Slider(1, 8, value=1, step=1, label="Batch Size")
                        
                        generate_btn = gr.Button("🎨 Generate", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        gallery = gr.Gallery(
                            label="Results",
                            columns=2,
                            height=600,
                            object_fit="contain",
                        )
                        gen_info = gr.Textbox(label="Info", interactive=False)
                
                # Events
                generate_btn.click(
                    fn=generate_image,
                    inputs=[model_dropdown, prompt_input, negative_input, steps_slider,
                            cfg_slider, width_slider, height_slider, seed_input, batch_slider],
                    outputs=[gallery, gen_info],
                )
                refresh_models_btn.click(
                    fn=lambda: gr.update(choices=get_model_choices()),
                    outputs=[model_dropdown],
                )
            
            # =============== TAB 2: MODELS ===============
            with gr.Tab("Models", id="models"):
                gr.Markdown("### Model Management")
                
                with gr.Row():
                    with gr.Column():
                        new_model_id = gr.Textbox(label="Model ID", placeholder="my-sd15")
                        pretrained_input = gr.Textbox(
                            label="Pretrained Path / HuggingFace ID",
                            placeholder="runwayml/stable-diffusion-v1-5",
                        )
                        model_config_input = gr.Code(
                            label="Model Config (JSON, optional)",
                            language="json",
                            value="",
                            lines=8,
                        )
                        
                        with gr.Row():
                            load_btn = gr.Button("📥 Load Model", variant="primary")
                            unload_btn = gr.Button("📤 Unload Model", variant="stop")
                    
                    with gr.Column():
                        model_status_text = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=2,
                        )
                        model_list_text = gr.Textbox(
                            label="Loaded Models",
                            value=get_model_list(),
                            interactive=False,
                            lines=10,
                        )
                        device_info = gr.Textbox(
                            label="Device",
                            value=_get_device_info(),
                            interactive=False,
                        )
                
                gr.Markdown("### Quick Load Presets")
                with gr.Row():
                    gr.Button("SD 1.5").click(
                        fn=lambda: ("sd15", "runwayml/stable-diffusion-v1-5", ""),
                        outputs=[new_model_id, pretrained_input, model_config_input],
                    )
                    gr.Button("SDXL").click(
                        fn=lambda: ("sdxl", "stabilityai/stable-diffusion-xl-base-1.0", ""),
                        outputs=[new_model_id, pretrained_input, model_config_input],
                    )
                    gr.Button("Flux Dev").click(
                        fn=lambda: ("flux", "black-forest-labs/FLUX.1-dev", ""),
                        outputs=[new_model_id, pretrained_input, model_config_input],
                    )
                
                # Events
                load_btn.click(
                    fn=load_model_ui,
                    inputs=[new_model_id, pretrained_input, model_config_input],
                    outputs=[model_status_text, model_list_text],
                )
                unload_btn.click(
                    fn=unload_model_ui,
                    inputs=[new_model_id],
                    outputs=[model_status_text, model_list_text],
                )
            
            # =============== TAB 3: TRAIN ===============
            with gr.Tab("Train", id="train"):
                gr.Markdown("### Train Model / Adapter")
                
                with gr.Row():
                    with gr.Column():
                        train_model_dropdown = gr.Dropdown(
                            label="Model to Train",
                            choices=get_model_choices(),
                        )
                        train_dataset_path = gr.Textbox(
                            label="Dataset Path",
                            placeholder="/path/to/images/",
                        )
                        train_mode_dropdown = gr.Dropdown(
                            label="Training Mode",
                            choices=["full", "adapter", "finetune"],
                            value="adapter",
                        )
                    
                    with gr.Column():
                        train_epochs = gr.Slider(1, 1000, value=10, step=1, label="Epochs")
                        train_batch = gr.Slider(1, 32, value=1, step=1, label="Batch Size")
                        train_lr = gr.Number(label="Learning Rate", value=1e-4)
                        train_precision = gr.Dropdown(
                            label="Mixed Precision",
                            choices=["no", "fp16", "bf16"],
                            value="fp16",
                        )
                        train_ema = gr.Checkbox(label="Use EMA", value=False)
                        train_save_every = gr.Slider(100, 5000, value=500, step=100, label="Save Every N Steps")
                
                train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                train_status = gr.Textbox(label="Training Status", interactive=False, lines=3)
                
                train_btn.click(
                    fn=start_training_ui,
                    inputs=[
                        train_model_dropdown, train_dataset_path, train_mode_dropdown,
                        train_epochs, train_batch, train_lr, train_precision,
                        train_ema, train_save_every,
                    ],
                    outputs=[train_status],
                )
            
            # =============== TAB 4: GRAPH EDITOR ===============
            with gr.Tab("Graph Editor", id="graph"):
                gr.Markdown("### ComputeGraph — Visual Pipeline Builder")
                gr.Markdown("Build custom diffusion pipelines by selecting templates, adding/replacing blocks, and wiring connections.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Template selector
                        def get_template_names():
                            try:
                                from ..core.graph.templates import list_templates
                                return list_templates()
                            except Exception:
                                return ["sd15_txt2img", "sdxl_txt2img", "sd3_txt2img", "flux_txt2img", "controlnet_txt2img"]
                        
                        template_dropdown = gr.Dropdown(
                            label="Pipeline Template",
                            choices=get_template_names(),
                            value="sd15_txt2img",
                            interactive=True,
                        )
                        load_template_btn = gr.Button("Load Template", variant="primary")
                        
                        gr.Markdown("#### Node Operations")
                        node_name_input = gr.Textbox(label="Node Name", placeholder="my_controlnet")
                        node_type_input = gr.Textbox(label="Block Type", placeholder="adapter/controlnet")
                        node_config_input = gr.Code(label="Node Config (JSON)", language="json", value="{}", lines=5)
                        
                        with gr.Row():
                            add_node_btn = gr.Button("Add Node")
                            replace_node_btn = gr.Button("Replace Node")
                            remove_node_btn = gr.Button("Remove Node", variant="stop")
                        
                        gr.Markdown("#### Connections")
                        src_node_input = gr.Textbox(label="Source (node.port)", placeholder="backbone.output")
                        dst_node_input = gr.Textbox(label="Destination (node.port)", placeholder="guidance.model_output")
                        connect_btn = gr.Button("Connect")
                    
                    with gr.Column(scale=2):
                        graph_mermaid = gr.Code(label="Graph Visualization (Mermaid)", language="markdown", lines=20)
                        graph_info = gr.Textbox(label="Graph Info", interactive=False, lines=5)
                        graph_yaml = gr.Code(label="Graph YAML", language="yaml", lines=15)
                
                # Graph state (stored as JSON)
                graph_state = gr.State(value=None)
                
                def load_template(template_name):
                    try:
                        from ..core.graph.graph import ComputeGraph
                        graph = ComputeGraph.from_template(template_name)
                        mermaid = graph.visualize()
                        info = repr(graph)
                        nodes_info = "\n".join([f"  {n}: {getattr(b, 'block_type', '?')}" for n, b in graph.nodes.items()])
                        return graph, mermaid, f"{info}\n\nNodes:\n{nodes_info}", ""
                    except Exception as e:
                        return None, "", f"Error: {e}", ""
                
                def add_node_to_graph(graph_obj, name, block_type, config_json):
                    if graph_obj is None:
                        return graph_obj, "No graph loaded", "", ""
                    try:
                        from ..core.block.builder import BlockBuilder
                        config = json.loads(config_json) if config_json.strip() else {}
                        config["type"] = block_type
                        block = BlockBuilder.build(config)
                        graph_obj.add_node(name, block)
                        return graph_obj, graph_obj.visualize(), repr(graph_obj), ""
                    except Exception as e:
                        return graph_obj, graph_obj.visualize(), f"Error: {e}", ""
                
                def remove_node_from_graph(graph_obj, name):
                    if graph_obj is None:
                        return graph_obj, "No graph loaded", "", ""
                    try:
                        graph_obj.remove_node(name)
                        return graph_obj, graph_obj.visualize(), repr(graph_obj), ""
                    except Exception as e:
                        return graph_obj, graph_obj.visualize(), f"Error: {e}", ""
                
                def connect_nodes(graph_obj, src_spec, dst_spec):
                    if graph_obj is None:
                        return graph_obj, "No graph loaded", "", ""
                    try:
                        src_node, src_port = src_spec.split(".", 1)
                        dst_node, dst_port = dst_spec.split(".", 1)
                        graph_obj.connect(src_node, src_port, dst_node, dst_port)
                        return graph_obj, graph_obj.visualize(), repr(graph_obj), ""
                    except Exception as e:
                        return graph_obj, graph_obj.visualize(), f"Error: {e}", ""
                
                load_template_btn.click(
                    fn=load_template,
                    inputs=[template_dropdown],
                    outputs=[graph_state, graph_mermaid, graph_info, graph_yaml],
                )
                add_node_btn.click(
                    fn=add_node_to_graph,
                    inputs=[graph_state, node_name_input, node_type_input, node_config_input],
                    outputs=[graph_state, graph_mermaid, graph_info, graph_yaml],
                )
                remove_node_btn.click(
                    fn=remove_node_from_graph,
                    inputs=[graph_state, node_name_input],
                    outputs=[graph_state, graph_mermaid, graph_info, graph_yaml],
                )
                connect_btn.click(
                    fn=connect_nodes,
                    inputs=[graph_state, src_node_input, dst_node_input],
                    outputs=[graph_state, graph_mermaid, graph_info, graph_yaml],
                )
            
            # =============== TAB 5: GRAPH TRAINING ===============
            with gr.Tab("Graph Training", id="graph_train"):
                gr.Markdown("### Train Any Node in Your Graph")
                gr.Markdown("Select which nodes to train — the rest will be frozen automatically.")
                
                with gr.Row():
                    with gr.Column():
                        gt_template = gr.Dropdown(
                            label="Pipeline Template",
                            choices=get_template_names(),
                            value="sd15_txt2img",
                        )
                        gt_train_nodes = gr.Textbox(
                            label="Train Nodes (comma-separated)",
                            placeholder="backbone, my_adapter",
                            value="backbone",
                        )
                        gt_dataset_path = gr.Textbox(label="Dataset Path", placeholder="/path/to/data/")
                        gt_loss = gr.Dropdown(label="Loss Type", choices=["epsilon", "velocity", "flow_matching", "x0"], value="epsilon")
                    
                    with gr.Column():
                        gt_epochs = gr.Slider(1, 1000, value=10, step=1, label="Epochs")
                        gt_batch = gr.Slider(1, 32, value=1, step=1, label="Batch Size")
                        gt_lr = gr.Number(label="Learning Rate", value=1e-4)
                        gt_optimizer = gr.Dropdown(label="Optimizer", choices=["adamw", "adam", "sgd"], value="adamw")
                        gt_precision = gr.Dropdown(label="Mixed Precision", choices=["no", "fp16", "bf16"], value="fp16")
                        gt_ema = gr.Checkbox(label="Use EMA", value=False)
                
                gt_start_btn = gr.Button("Start Graph Training", variant="primary", size="lg")
                gt_status = gr.Textbox(label="Status", interactive=False, lines=3)
                
                def start_graph_training(template, train_nodes_str, dataset_path, loss_type, epochs, batch, lr, optimizer, precision, use_ema):
                    try:
                        from ..core.graph.graph import ComputeGraph
                        from ..training.graph_trainer import GraphTrainer, GraphTrainingConfig
                        from ..training.data import ImageFolderSource
                        
                        graph = ComputeGraph.from_template(template)
                        train_nodes = [n.strip() for n in train_nodes_str.split(",")]
                        
                        config = GraphTrainingConfig(
                            num_epochs=int(epochs),
                            batch_size=int(batch),
                            learning_rate=float(lr),
                            optimizer=optimizer,
                            mixed_precision=precision,
                            use_ema=use_ema,
                            loss_type=loss_type,
                        )
                        
                        trainer = GraphTrainer(graph=graph, train_nodes=train_nodes, config=config)
                        dataset = ImageFolderSource(dataset_path)
                        
                        import threading
                        def run():
                            try:
                                trainer.train(dataset)
                            except Exception as e:
                                print(f"Graph training error: {e}")
                        
                        thread = threading.Thread(target=run, daemon=True)
                        thread.start()
                        
                        return f"Graph training started | Template: {template} | Train nodes: {train_nodes} | Epochs: {epochs}"
                    except Exception as e:
                        return f"Error: {e}"
                
                gt_start_btn.click(
                    fn=start_graph_training,
                    inputs=[gt_template, gt_train_nodes, gt_dataset_path, gt_loss, gt_epochs, gt_batch, gt_lr, gt_optimizer, gt_precision, gt_ema],
                    outputs=[gt_status],
                )
            
            # =============== TAB 6: BLOCKS ===============
            with gr.Tab("Blocks", id="blocks"):
                gr.Markdown("### Registered Blocks (Lego Bricks)")
                gr.Markdown("All available blocks in the registry. You can use these to build custom models and graphs.")
                
                def get_blocks_info() -> str:
                    from ..core.block.registry import list_blocks
                    blocks = list_blocks()
                    if not blocks:
                        return "No blocks registered"
                    
                    lines = []
                    categories = {}
                    for key, cls in sorted(blocks.items()):
                        cat = key.split("/")[0] if "/" in key else "other"
                        categories.setdefault(cat, []).append((key, cls))
                    
                    for cat, items in sorted(categories.items()):
                        lines.append(f"\n### {cat.upper()}")
                        for key, cls in items:
                            doc = (cls.__doc__ or "").split("\n")[0].strip()
                            # Show ports if available
                            ports = ""
                            try:
                                io_ports = cls.declare_io()
                                if io_ports:
                                    in_ports = [p.name for p in io_ports.values() if p.direction == "input"]
                                    out_ports = [p.name for p in io_ports.values() if p.direction == "output"]
                                    ports = f" | In: {in_ports} | Out: {out_ports}"
                            except Exception:
                                pass
                            lines.append(f"- `{key}` — {doc}{ports}")
                    
                    return "\n".join(lines)
                
                blocks_display = gr.Markdown(value=get_blocks_info())
                gr.Button("Refresh").click(fn=get_blocks_info, outputs=[blocks_display])
            
            # =============== TAB 7: DEPLOY ===============
            with gr.Tab("Deploy", id="deploy"):
                gr.Markdown("### One-Click Deployment")
                
                with gr.Row():
                    with gr.Column():
                        deploy_target = gr.Dropdown(
                            label="Deploy Target",
                            choices=["Local Server", "RunPod Serverless", "Vast.ai"],
                            value="Local Server",
                        )
                        deploy_graph_template = gr.Dropdown(
                            label="Pipeline Template",
                            choices=get_template_names(),
                            value="sd15_txt2img",
                        )
                        deploy_port = gr.Number(label="Port (Local)", value=8000, precision=0)
                        deploy_api_key = gr.Textbox(label="API Key (for cloud)", type="password", placeholder="your-api-key")
                        deploy_gpu_type = gr.Textbox(label="GPU Type (cloud)", placeholder="RTX_4090")
                    
                    with gr.Column():
                        deploy_btn = gr.Button("Deploy", variant="primary", size="lg")
                        deploy_status = gr.Textbox(label="Deployment Status", interactive=False, lines=5)
                
                def deploy_graph(target, template, port, api_key, gpu_type):
                    return f"Deployment configured: {target} | Template: {template} | Port: {port}"
                
                deploy_btn.click(
                    fn=deploy_graph,
                    inputs=[deploy_target, deploy_graph_template, deploy_port, deploy_api_key, deploy_gpu_type],
                    outputs=[deploy_status],
                )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 2em; color: #888; font-size: 0.85em;">
            YggDrasil v2.0 — True Lego Constructor for Diffusion Models
        </div>
        """)
    
    return demo
