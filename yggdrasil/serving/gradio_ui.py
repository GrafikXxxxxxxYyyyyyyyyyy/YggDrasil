# yggdrasil/serving/gradio_ui.py
"""Gradio UI для YggDrasil — единый интерфейс для любых диффузионных моделей (image, video, audio).

Особенности:
- Выбор модальности (Изображение / Видео / Аудио) и шаблона пайплайна
- Динамические входы по графу (prompt, control_image, num_frames и т.д.)
- Пресеты разрешений и параметров, семя, батч, скачивание результата
- Информация об устройстве и понятные сообщения об ошибках
"""
from __future__ import annotations

import io
import time
import random
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from .schema import ServerConfig


# ==================== HELPERS ====================

def _get_device_info() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
        return f"CUDA: {name} ({mem:.1f} GB)"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "Apple MPS"
    return "CPU"


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _tensor_to_pil_list(tensor: torch.Tensor) -> List[Image.Image]:
    """Тензор [B, C, H, W] в [-1,1] или [0,1] → список PIL."""
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return []
    img = tensor.detach().cpu().float()
    if img.min() < -0.01 or img.max() > 1.01:
        img = (img / 2 + 0.5).clamp(0, 1)
    else:
        img = img.clamp(0, 1)
    images = []
    for i in range(img.shape[0]):
        arr = (img[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        images.append(Image.fromarray(arr))
    return images


def _video_tensor_to_file(tensor: torch.Tensor, fps: float = 8.0) -> Optional[str]:
    """Тензор видео [B,C,T,H,W] или [C,T,H,W] → временный файл .mp4."""
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return None
    try:
        import tempfile
        t = tensor.detach().cpu().float()
        if t.min() < -0.01 or t.max() > 1.01:
            t = (t / 2 + 0.5).clamp(0, 1)
        if t.dim() == 5:
            t = t[0]
        # [C, T, H, W] → frames (T, H, W, C)
        t = t.permute(1, 2, 3, 0).numpy()
        t = (t * 255).clip(0, 255).astype(np.uint8)
        path = tempfile.mktemp(suffix=".mp4")
        try:
            import imageio
            imageio.mimwrite(path, t, fps=fps)
            return path
        except ImportError:
            return None
    except Exception:
        return None


def _audio_tensor_to_file(tensor: torch.Tensor, sr: int = 44100) -> Optional[Tuple[int, np.ndarray]]:
    """Тензор аудио [B, C, T] или [C, T] → (sample_rate, np.ndarray)."""
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return None
    try:
        a = tensor.detach().cpu().float().numpy()
        if a.ndim == 3:
            a = a[0]
        if a.ndim == 2:
            a = a.mean(axis=0)
        a = np.clip(a, -1, 1).astype(np.float32)
        return (sr, a)
    except Exception:
        return None


def _get_templates_by_modality() -> Dict[str, List[Tuple[str, str]]]:
    """Возвращает {modality: [(template_id, description), ...]}."""
    try:
        from yggdrasil.pipeline import Pipeline
        available = Pipeline.list_available()
    except Exception:
        available = {}
    result = {"image": [], "video": [], "audio": []}
    for name, info in available.items():
        desc = (info.get("description") or name).strip().split("\n")[0][:80]
        mod = info.get("modality", "image")
        if mod not in result:
            result[mod] = []
        result[mod].append((name, desc))
    for mod in result:
        result[mod].sort(key=lambda x: x[0])
    if not any(result.values()):
        result["image"] = [("sd15_txt2img", "SD 1.5 Text-to-Image")]
    return result


# ==================== MAIN UI ====================

def create_ui(
    manager: Optional[Any] = None,
    config: Optional[ServerConfig] = None,
    share: bool = False,
) -> "gr.Blocks":
    """Создать единый Gradio интерфейс для любых диффузионных моделей."""
    import gradio as gr

    templates_by_mod = _get_templates_by_modality()
    device = _best_device()
    device_info = _get_device_info()

    # ---------- Generation logic ----------
    def run_generation(
        modality: str,
        template_name: str,
        prompt: str,
        negative_prompt: str,
        num_steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        num_frames: int,
        seed: int,
        batch_size: int,
        control_image: Optional[Any],
        ip_image: Optional[Any],
        source_image: Optional[Any],
        pipeline_state: Optional[Tuple[str, Any]],
    ) -> Tuple[
        Optional[List[Image.Image]],
        Optional[str],
        Optional[Tuple[int, np.ndarray]],
        str,
        Optional[Tuple[str, Any]],
    ]:
        """Запуск генерации. Возвращает (images, video_path, audio_tuple, info, new_pipeline_state)."""
        if not template_name or template_name not in [t[0] for t in templates_by_mod.get(modality, [])]:
            return [], None, None, "Выберите шаблон пайплайна.", pipeline_state

        try:
            from yggdrasil.pipeline import Pipeline
        except ImportError as e:
            return [], None, None, f"Ошибка импорта: {e}", pipeline_state

        # Reuse pipeline if same template
        pipe = None
        if pipeline_state and pipeline_state[0] == template_name:
            pipe = pipeline_state[1]
        if pipe is None:
            try:
                pipe = Pipeline.from_template(template_name, device=device)
            except Exception as e:
                return [], None, None, f"Не удалось загрузить пайплайн: {e}", pipeline_state
            pipeline_state = (template_name, pipe)

        def _pil_to_tensor(pil_img) -> torch.Tensor:
            if pil_img is None:
                return None
            if isinstance(pil_img, dict) and "image" in pil_img:
                pil_img = pil_img["image"]
            arr = np.array(pil_img).astype(np.float32) / 255.0
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            return t

        actual_seed = int(seed) if seed >= 0 else random.randint(0, 2**32 - 1)
        kwargs = {
            "prompt": prompt or "a beautiful scene",
            "negative_prompt": negative_prompt or "",
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": actual_seed,
            "batch_size": min(max(1, batch_size), 8),
        }
        if modality == "video":
            kwargs["num_frames"] = num_frames
        if control_image is not None:
            kwargs["control_image"] = _pil_to_tensor(control_image)
        if ip_image is not None:
            kwargs["ip_image"] = ip_image if isinstance(ip_image, dict) else {"image": ip_image}
        if source_image is not None and modality in ("video", "image"):
            kwargs["source_image"] = _pil_to_tensor(source_image)

        start = time.time()
        try:
            out = pipe(**kwargs)
        except Exception as e:
            return [], None, None, f"Ошибка генерации: {e}", pipeline_state
        elapsed = time.time() - start

        images = None
        video_path = None
        audio_data = None
        if out.images:
            images = out.images
        if getattr(out, "video", None) is not None:
            video_path = _video_tensor_to_file(out.video)
        if getattr(out, "audio", None) is not None:
            audio_data = _audio_tensor_to_file(out.audio)

        info = f"Seed: {actual_seed} | Steps: {num_steps} | CFG: {guidance_scale} | {elapsed:.1f}s | {device_info}"
        return images or [], video_path, audio_data, info, pipeline_state

    def update_template_choices(modality: str):
        choices = templates_by_mod.get(modality, [])
        return gr.update(choices=[t[0] for t in choices], value=choices[0][0] if choices else None)

    # ---------- Build UI ----------
    with gr.Blocks(
        title="YggDrasil — Universal Diffusion",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
        ),
        css="""
        .hero { text-align: center; margin-bottom: 0.5em; font-size: 1.8em; }
        .sub { text-align: center; color: #64748b; margin-bottom: 1.2em; }
        .preset-btn { min-width: 4em; }
        .footer { text-align: center; margin-top: 1.5em; color: #94a3b8; font-size: 0.9em; }
        """,
    ) as demo:

        gr.HTML('<h1 class="hero">🌳 YggDrasil</h1><p class="sub">Единый интерфейс для генерации изображений, видео и звука</p>')

        pipeline_state = gr.State(value=None)

        with gr.Tabs():
            # ========== TAB: GENERATE ==========
            with gr.Tab("🎨 Генерация", id="gen"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Модель и модальность")
                        modality_radio = gr.Radio(
                            choices=[("Изображение", "image"), ("Видео", "video"), ("Аудио", "audio")],
                            value="image",
                            label="Тип",
                            elem_id="modality",
                        )
                        template_dropdown = gr.Dropdown(
                            label="Пайплайн",
                            choices=[t[0] for t in templates_by_mod["image"]],
                            value=templates_by_mod["image"][0][0] if templates_by_mod["image"] else None,
                            interactive=True,
                        )
                        num_frames_num = gr.Slider(4, 64, value=16, step=1, label="Кадров (видео)", visible=False)
                        def on_modality(m):
                            choices = templates_by_mod.get(m, [])
                            vis = m == "video"
                            return (
                                gr.update(choices=[t[0] for t in choices], value=choices[0][0] if choices else None),
                                gr.update(visible=vis),
                            )
                        modality_radio.change(
                            fn=on_modality,
                            inputs=[modality_radio],
                            outputs=[template_dropdown, num_frames_num],
                        )

                        gr.Markdown("### Текст и параметры")
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="описание того, что вы хотите получить...",
                            lines=3,
                        )
                        negative_input = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="размыто, низкое качество...",
                            lines=2,
                        )
                        with gr.Row():
                            steps_num = gr.Slider(1, 150, value=28, step=1, label="Шаги")
                            cfg_num = gr.Slider(0.5, 30, value=7.5, step=0.5, label="CFG")
                        with gr.Row():
                            width_num = gr.Slider(128, 2048, value=512, step=64, label="Ширина")
                            height_num = gr.Slider(128, 2048, value=512, step=64, label="Высота")
                        with gr.Row():
                            seed_num = gr.Number(label="Seed (-1 = случайный)", value=-1, precision=0)
                            seed_random_btn = gr.Button("🎲 Случайный", size="sm")
                            batch_num = gr.Slider(1, 8, value=1, step=1, label="Батч")
                        seed_random_btn.click(lambda: -1, outputs=[seed_num])

                        gr.Markdown("#### Пресеты")
                        with gr.Row():
                            gr.Button("512×512").click(lambda: (512, 512), outputs=[width_num, height_num])
                            gr.Button("768×768").click(lambda: (768, 768), outputs=[width_num, height_num])
                            gr.Button("1024×1024").click(lambda: (1024, 1024), outputs=[width_num, height_num])
                            gr.Button("Быстро (20 шагов)").click(lambda: 20, outputs=[steps_num])
                            gr.Button("Качество (40 шагов)").click(lambda: 40, outputs=[steps_num])
                        gr.Markdown("#### Адаптеры (опционально)")
                        control_image_in = gr.Image(label="Control (depth/canny)", type="pil")
                        ip_image_in = gr.Image(label="IP-Adapter изображение", type="pil")
                        source_image_in = gr.Image(label="Исходное изображение (img2vid)", type="pil")

                        gen_btn = gr.Button("🚀 Сгенерировать", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gr.Markdown("### Результат")
                        out_gallery = gr.Gallery(
                            label="Изображения",
                            columns=3,
                            height=500,
                            object_fit="contain",
                        )
                        out_video = gr.Video(label="Видео", visible=False)
                        out_audio = gr.Audio(label="Аудио", visible=False)
                        gen_info = gr.Textbox(label="Инфо", interactive=False)
                        download_btn = gr.DownloadButton(label="Скачать первый результат", visible=True)

                def run_and_show(
                    mod, tpl, prompt, neg, steps, cfg, w, h, nf, seed, batch,
                    ctrl_img, ip_img, src_img, state,
                ):
                    images, video_path, audio_data, info, new_state = run_generation(
                        mod, tpl, prompt, neg, steps, cfg, w, h, nf, seed, batch,
                        ctrl_img, ip_img, src_img, state,
                    )
                    vis_img = bool(images and len(images) > 0)
                    vis_vid = video_path is not None
                    vis_aud = audio_data is not None
                    # Download: first image as bytes or video path
                    download_file = None
                    if images and len(images) > 0:
                        buf = io.BytesIO()
                        images[0].save(buf, format="PNG")
                        buf.seek(0)
                        download_file = (buf.getvalue(), "yggdrasil_output.png")
                    elif video_path:
                        download_file = video_path
                    return (
                        gr.update(value=images or [], visible=vis_img),
                        gr.update(value=video_path, visible=vis_vid),
                        gr.update(value=audio_data, visible=vis_aud),
                        info,
                        new_state,
                        download_file,
                    )

                gen_btn.click(
                    fn=run_and_show,
                    inputs=[
                        modality_radio, template_dropdown,
                        prompt_input, negative_input,
                        steps_num, cfg_num, width_num, height_num, num_frames_num,
                        seed_num, batch_num,
                        control_image_in, ip_image_in, source_image_in,
                        pipeline_state,
                    ],
                    outputs=[
                        out_gallery,
                        out_video,
                        out_audio,
                        gen_info,
                        pipeline_state,
                        download_btn,
                    ],
                )

            # ========== TAB: PIPELINES ==========
            with gr.Tab("📦 Пайплайны", id="pipelines"):
                gr.Markdown("### Доступные шаблоны по типу")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Изображения")
                        img_list = "\n".join(f"- **{t[0]}** — {t[1]}" for t in templates_by_mod.get("image", [])[:20])
                        gr.Markdown(img_list or "Нет шаблонов")
                    with gr.Column():
                        gr.Markdown("#### Видео")
                        vid_list = "\n".join(f"- **{t[0]}** — {t[1]}" for t in templates_by_mod.get("video", [])[:20])
                        gr.Markdown(vid_list or "Нет шаблонов")
                    with gr.Column():
                        gr.Markdown("#### Аудио")
                        aud_list = "\n".join(f"- **{t[0]}** — {t[1]}" for t in templates_by_mod.get("audio", [])[:20])
                        gr.Markdown(aud_list or "Нет шаблонов")
                gr.Markdown("Выберите шаблон во вкладке **Генерация** и нажмите Сгенерировать. Модели подгружаются при первом запуске.")

            # ========== TAB: BLOCKS ==========
            with gr.Tab("🧱 Блоки", id="blocks"):
                gr.Markdown("### Зарегистрированные блоки (Lego)")
                def get_blocks_md():
                    try:
                        from yggdrasil.core.block.registry import list_blocks
                        blocks = list_blocks()
                        if not blocks:
                            return "Нет зарегистрированных блоков."
                        by_cat = {}
                        for k, cls in sorted(blocks.items()):
                            cat = k.split("/")[0] if "/" in k else "other"
                            by_cat.setdefault(cat, []).append((k, cls))
                        lines = []
                        for cat, items in sorted(by_cat.items()):
                            lines.append(f"\n### {cat.upper()}")
                            for k, cls in items:
                                doc = (cls.__doc__ or "").split("\n")[0].strip()[:70]
                                lines.append(f"- `{k}` — {doc}")
                        return "\n".join(lines)
                    except Exception as e:
                        return f"Ошибка: {e}"
                blocks_md = gr.Markdown(value=get_blocks_md())
                gr.Button("Обновить").click(fn=get_blocks_md, outputs=[blocks_md])

            # ========== TAB: TRAIN ==========
            with gr.Tab("🎓 Обучение", id="train"):
                gr.Markdown("### Обучение адаптера / дообучение графа")
                with gr.Row():
                    with gr.Column():
                        train_template = gr.Dropdown(
                            label="Шаблон графа",
                            choices=[t[0] for t in (templates_by_mod["image"] + templates_by_mod.get("video", []) + templates_by_mod.get("audio", []))],
                            value=templates_by_mod["image"][0][0] if templates_by_mod["image"] else None,
                        )
                        train_nodes = gr.Textbox(label="Обучаемые узлы (через запятую)", value="lora_adapter", placeholder="backbone, lora_adapter")
                        train_data = gr.Textbox(label="Путь к данным", placeholder="/path/to/images/")
                        train_epochs = gr.Slider(1, 500, value=10, step=1, label="Эпохи")
                        train_lr = gr.Number(label="Learning rate", value=1e-4)
                    with gr.Column():
                        train_btn = gr.Button("Запустить обучение", variant="primary")
                        train_status = gr.Textbox(label="Статус", interactive=False, lines=5)
                def start_train(tpl, nodes, data, epochs, lr):
                    if not data or not Path(data).exists():
                        return f"Путь не найден: {data}"
                    try:
                        from yggdrasil.core.graph.graph import ComputeGraph
                        from yggdrasil.training.graph_trainer import GraphTrainer, GraphTrainingConfig
                        from yggdrasil.training.data import ImageFolderSource
                        g = ComputeGraph.from_template(tpl)
                        nlist = [n.strip() for n in nodes.split(",") if n.strip()]
                        cfg = GraphTrainingConfig(num_epochs=int(epochs), batch_size=1, learning_rate=float(lr))
                        trainer = GraphTrainer(graph=g, train_nodes=nlist, config=cfg)
                        ds = ImageFolderSource(data)
                        import threading
                        def run():
                            try:
                                trainer.train(ds)
                            except Exception as e:
                                print(f"Train error: {e}")
                        threading.Thread(target=run, daemon=True).start()
                        return f"Обучение запущено: {tpl}, узлы {nlist}, эпох {epochs}"
                    except Exception as e:
                        return f"Ошибка: {e}"
                train_btn.click(
                    fn=start_train,
                    inputs=[train_template, train_nodes, train_data, train_epochs, train_lr],
                    outputs=[train_status],
                )

        gr.HTML(f'<div class="footer">YggDrasil — Lego для диффузии · {device_info}</div>')

    return demo
