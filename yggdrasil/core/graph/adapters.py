# yggdrasil/core/graph/adapters.py
"""Подключение адаптеров (ControlNet, IP-Adapter и др.) к любому графу.

Любой адаптер можно добавить к графу с denoise_loop и backbone, поддерживающим
adapter_features. Исходные данные для адаптера можно передавать как новый вход
графа или по ссылке на выход другого узла (node_name, port_name).
"""
from __future__ import annotations

import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union

from .graph import ComputeGraph
from ..block.builder import BlockBuilder


def _cross_attention_dim_from_model_config(config: Any) -> Optional[int]:
    """Универсально прочитать cross-attention размерность из config модели (UNet, Transformer, DiT и т.д.)."""
    if config is None:
        return None
    dim = (
        getattr(config, "cross_attention_dim", None)
        or getattr(config, "encoder_hidden_size", None)
        or getattr(config, "joint_attention_dim", None)
        or getattr(config, "hidden_size", None)
        or getattr(config, "text_embed_dim", None)
    )
    return int(dim) if dim is not None and isinstance(dim, (int, float)) else None


def set_ip_adapter_processors_on_unet(unet: Any, scale: float = 0.6, num_tokens: Tuple[int, ...] = (4,)) -> None:
    """Set diffusers UNet attention processors to IP-Adapter type so (text_embeds, image_embeds) are used.

    Call this when the graph has an IP-Adapter node and the backbone will receive image_prompt_embeds.
    Without this, the UNet would ignore the image part of encoder_hidden_states.
    """
    from diffusers.models.attention_processor import IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0

    attn_procs = {}
    cross_attention_dim = _cross_attention_dim_from_model_config(getattr(unet, "config", None)) or 2048
    block_out_channels = list(getattr(unet.config, "block_out_channels", [320, 640, 1280, 1280]))

    for name in unet.attn_processors.keys():
        is_cross = not name.endswith("attn1.processor") and "motion_modules" not in name
        if not is_cross:
            attn_procs[name] = unet.attn_processors[name].__class__()
            continue
        if name.startswith("mid_block"):
            hidden_size = block_out_channels[-1]
        elif name.startswith("up_blocks"):
            # name is e.g. "up_blocks.0.attentions.0.transformer_blocks.0.attn2.processor"
            block_id = int(name.split(".", 2)[1])
            hidden_size = list(reversed(block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name.split(".", 2)[1])
            hidden_size = block_out_channels[block_id]
        else:
            attn_procs[name] = unet.attn_processors[name].__class__()
            continue
        processor_class = (
            IPAdapterAttnProcessor2_0
            if hasattr(F, "scaled_dot_product_attention")
            else IPAdapterAttnProcessor
        )
        attn_procs[name] = processor_class(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            num_tokens=num_tokens,
            scale=scale,
        )
    unet.set_attn_processor(attn_procs)
    # New processors are created in float32; UNet is often float16. Align dtype/device so matmuls don't fail.
    # Skip when UNet is still on meta (lazy load): .to() would raise "Cannot copy out of meta tensor".
    # Processors will be moved when the graph is moved to device (graph.to(device)).
    first_param = next(unet.parameters(), None)
    if first_param is not None and unet.attn_processors:
        _device, _dtype = first_param.device, first_param.dtype
        if str(_device) == "meta" or getattr(_device, "type", None) == "meta":
            pass
        else:
            try:
                for _proc in unet.attn_processors.values():
                    if hasattr(_proc, "to"):
                        _proc.to(device=_device, dtype=_dtype)
            except NotImplementedError as e:
                if "meta" in str(e) or "Cannot copy out of meta" in str(e):
                    pass
                else:
                    raise


def ensure_ip_adapter_processors_on_device(unet: Any) -> None:
    """Перенести процессоры IP-Adapter на устройство и dtype UNet (после graph.to(device)).

    Вызывать после переноса графа на устройство: процессоры могли не перенестись, если UNet
    при материализации был на meta. Без этого генерация может давать серый/поломанный вывод.
    """
    if not getattr(unet, "attn_processors", None):
        return
    first_param = next(unet.parameters(), None)
    if first_param is None:
        return
    _device, _dtype = first_param.device, first_param.dtype
    if str(_device) == "meta" or getattr(_device, "type", None) == "meta":
        return
    try:
        for _proc in unet.attn_processors.values():
            if hasattr(_proc, "to"):
                _proc.to(device=_device, dtype=_dtype)
    except NotImplementedError:
        pass


def set_ip_adapter_scale_on_unet(unet: Any, scale: float) -> None:
    """Set IP-Adapter strength (0..1+) on all IP-Adapter attention processors of a diffusers UNet.

    Call this before each generation if you want to change strength per call (e.g. pipe(..., ip_adapter_scale=0.9)).
    """
    from diffusers.models.attention_processor import IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0

    if not getattr(unet, "attn_processors", None):
        return
    for proc in unet.attn_processors.values():
        if isinstance(proc, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)):
            n = len(proc.scale) if isinstance(proc.scale, (list, tuple)) else 1
            proc.scale = [scale] * n


# Тип "источник по ссылке": (имя_узла, имя_порта)
SourceRef = Tuple[str, str]


def get_cross_attention_dim_from_graph(graph: ComputeGraph) -> Optional[int]:
    """Универсально получить cross_attention_dim из уже собранного backbone в графе.

    Ищет узел-цикл (loop), внутри — backbone, затем модель (unet / transformer / _model)
    и читает размерность из config. Подходит для любой диффузионной модели (SD 1.5, SDXL,
    SD3, FLUX, WAN, Animate Diffusion, LDM и т.д.), если у модели в config есть одна из
    типичных полей.

    Returns:
        cross_attention_dim или None, если backbone ещё не собран или атрибут не найден.
    """
    for _node_name, block in (graph.nodes or {}).items():
        inner = None
        if getattr(block, "block_type", "").startswith("loop/"):
            inner = getattr(block, "graph", None) or (
                getattr(block, "_loop", None) and getattr(block._loop, "graph", None)
            )
        if inner is None:
            continue
        inodes = getattr(inner, "nodes", {}) or {}
        backbone = inodes.get("backbone")
        if backbone is None:
            for _n, b in inodes.items():
                if getattr(b, "block_type", "").startswith("backbone/"):
                    backbone = b
                    break
        if backbone is None:
            continue
        # Модель: unet (SD/UNet2D), transformer (FLUX, SD3), _model (общий запас)
        model = getattr(backbone, "unet", None) or getattr(backbone, "transformer", None) or getattr(backbone, "_model", None)
        if model is None:
            continue
        dim = _cross_attention_dim_from_model_config(getattr(model, "config", None))
        if dim is not None:
            return dim
    return None


def get_controlnet_input_mapping(graph: ComputeGraph) -> Dict[str, str]:
    """Маппинг control_type → имя входа графа для нескольких ControlNet (§11.3 A2).

    Обходит внутренний граф цикла, находит узлы adapter/controlnet и для каждого
    возвращает пару (control_type блока, имя graph_input: control_image или control_image_<node_name>).
    Единая точка для пайплайна и UI: передача по control_type (depth, canny, …) или по имени входа.

    Returns:
        Словарь {control_type: graph_input_name}, например {"depth": "control_image_controlnet_depth", "canny": "control_image_controlnet_canny"}.
    """
    out: Dict[str, str] = {}
    for _node_name, block in (graph.nodes or {}).items():
        inner = None
        if getattr(block, "block_type", "").startswith("loop/"):
            inner = getattr(block, "graph", None) or (
                getattr(block, "_loop", None) and getattr(block._loop, "graph", None)
            )
        if inner is None:
            continue
        graph_inputs = getattr(inner, "graph_inputs", {}) or {}
        nodes = getattr(inner, "nodes", {}) or {}
        for inp_name, targets in graph_inputs.items():
            if inp_name != "control_image" and not inp_name.startswith("control_image_"):
                continue
            for (node_name, _port) in targets:
                if node_name in nodes:
                    blk = nodes[node_name]
                    if getattr(blk, "block_type", None) == "adapter/controlnet":
                        ct = getattr(blk, "control_type", None) or inp_name
                        out[ct] = inp_name
                        break
        break
    return out


# Backbone types that accept adapter_features (ControlNet/T2I residuals)
_BACKBONE_TYPES_WITH_ADAPTER = (
    "backbone/unet2d_condition",
    "backbone/unet2d_batched_cfg",
    "backbone/unet3d_condition",
)


def _backbone_supports_adapters(backbone: Any) -> bool:
    """True if the backbone has adapter_features input (ControlNet/T2I)."""
    bt = getattr(backbone, "block_type", "")
    return bt in _BACKBONE_TYPES_WITH_ADAPTER


def add_controlnet_to_graph(
    graph: ComputeGraph,
    controlnet_pretrained: str = "lllyasviel/control_v11p_sd15_canny",
    conditioning_scale: float = 1.0,
    fp16: bool = True,
    control_image_source: Optional[SourceRef] = None,
    denoise_loop_node: str = "denoise_loop",
    controlnet_block: Optional[Any] = None,
    controlnet_node_name: Optional[str] = None,
    **controlnet_kwargs: Any,
) -> ComputeGraph:
    """Добавить ControlNet как отдельный блок к существующему графу.

    Граф должен содержать узел ``denoise_loop`` (LoopSubGraph). Внутренний
    шаг должен содержать backbone с поддержкой ``adapter_features``
    (например ``backbone/unet2d_condition``). Подходит для SD 1.5 и SDXL
    (для SDXL укажите controlnet_pretrained для SDXL, напр. diffusers/controlnet-canny-sdxl-1.0).

    Args:
        graph: Исходный граф (sd15_txt2img, sdxl_txt2img, sd15_txt2img_nobatch и т.д.).
        controlnet_pretrained: HF model ID ControlNet (SD 1.5 или SDXL).
        conditioning_scale: Сила контроля (0..1).
        fp16: Использовать float16 для ControlNet.
        control_image_source: Если задано (node_name, port_name) — control_image берётся
            с выхода указанного узла (по ссылке). Иначе добавляется новый вход графа control_image.
        **controlnet_kwargs: Доп. параметры для блока adapter/controlnet.

    Returns:
        Тот же граф (in-place) с добавленным ControlNet.

    Example::
        # Внешний вход
        add_controlnet_to_graph(graph, controlnet_pretrained="lllyasviel/control_v11p_sd15_canny")
        out = graph.execute(prompt="a cat", control_image=my_canny_tensor, ...)

        # По ссылке: control_image с выхода препроцессора
        graph.add_node("canny", canny_block)
        graph.connect("source_image", "canny", "image")  # или expose_input
        add_controlnet_to_graph(graph, control_image_source=("canny", "output"))
    """
    if denoise_loop_node not in graph.nodes:
        raise ValueError(f"Graph must have node '{denoise_loop_node}' (LoopSubGraph)")

    loop = graph.nodes[denoise_loop_node]
    inner = getattr(loop, "graph", None) or (
        getattr(loop, "_loop", None) and getattr(loop._loop, "graph", None)
    )
    if inner is None:
        raise ValueError(f"'{denoise_loop_node}' must have an inner graph (.graph or ._loop.graph)")

    if "backbone" not in inner.nodes:
        raise ValueError("Inner graph must have a 'backbone' node")

    backbone = inner.nodes["backbone"]
    if not _backbone_supports_adapters(backbone):
        return graph  # skip: DiT/WAN etc. do not support adapter_features

    # Имя узла: задано пользователем (controlnet_node_name) или controlnet, controlnet_1, ...
    if controlnet_node_name:
        node_name = controlnet_node_name
        if node_name in inner.nodes:
            node_name = f"{controlnet_node_name}_{len(inner.nodes)}"
    else:
        prefix = "controlnet_"
        existing = [n for n in inner.nodes if n == "controlnet" or (n.startswith(prefix) and n[len(prefix):].isdigit())]
        if not existing:
            node_name = "controlnet"
        else:
            indices = [0]
            for n in existing:
                if n.startswith(prefix) and n[len(prefix):].isdigit():
                    indices.append(int(n[len(prefix):]))
            node_name = f"controlnet_{max(indices) + 1}" if any(n != "controlnet" for n in existing) else "controlnet_1"

    if controlnet_block is not None:
        controlnet = controlnet_block
    else:
        config = {
            "type": "adapter/controlnet",
            "pretrained": controlnet_pretrained,
            "conditioning_scale": conditioning_scale,
            "fp16": fp16,
            **controlnet_kwargs,
        }
        controlnet = BlockBuilder.build(config)
    inner.add_node(node_name, controlnet)

    # Unique graph input per ControlNet: control_image (first, backward compat), control_image_<node_name> (others)
    control_type = getattr(controlnet, "control_type", "canny")
    if node_name == "controlnet":
        input_name = "control_image"  # backward compat: single ControlNet
    else:
        # e.g. control_image_controlnet_1 so pipeline can pass by list order or dict by key
        input_name = f"control_image_{node_name}"

    inner.expose_input(input_name, node_name, "control_image")
    inner.expose_input("condition", node_name, "encoder_hidden_states")
    inner.expose_input("latents", node_name, "sample")
    inner.expose_input("timestep", node_name, "timestep")
    inner.connect(node_name, "output", "backbone", "adapter_features")

    if control_image_source is not None:
        src_node, src_port = control_image_source
        if src_node not in graph.nodes:
            raise ValueError(f"control_image_source node '{src_node}' not found in graph")
        graph.connect(src_node, src_port, denoise_loop_node, input_name)
    else:
        graph.expose_input(input_name, denoise_loop_node, input_name)
    return graph


def add_ip_adapter_to_graph(
    graph: ComputeGraph,
    ip_adapter_scale: float = 0.6,
    image_source: Optional[SourceRef] = None,
    image_encoder_pretrained: Optional[str] = None,
    cross_attention_dim: Optional[int] = None,
    **ip_adapter_kwargs: Any,
) -> ComputeGraph:
    """Добавить IP-Adapter к графу: блок кодирования изображения + IP-Adapter.

    Добавляет узел image_encoder (CLIP vision) и adapter/ip_adapter. Входное
    изображение передаётся входом графа ``ip_image`` или по ссылке
    (image_source=(node_name, port_name)).

    Поддержка нескольких изображений (multi-image):
        - В pipeline: ip_image может быть один (Path/URL/PIL/tensor), список таких,
          или словарь (словарь образов: ключ -> изображение).
        - Граф ожидает ip_image в виде dict: {"image": один} или {"images": [img1, img2, ...]}.
        - Энкодер выдаёт батч эмбеддингов; IP-Adapter объединяет их в один conditioning
          (multi_image_mode: "mean" или "first").

    Args:
        graph: Граф с denoise_loop и backbone.
        ip_adapter_scale: Сила влияния image prompt.
        image_source: Если задано (node_name, port_name) — изображение берётся
            с выхода узла. Иначе добавляется вход графа ip_image.
        image_encoder_pretrained: HF model ID для CLIP image encoder (по умолчанию из контекста).
        cross_attention_dim: Cross-attention dim модели. Если None — берётся из уже собранного backbone в графе (универсально для любой модели: SD, SDXL, SD3, FLUX, WAN и т.д.).
        **ip_adapter_kwargs: Доп. параметры для adapter/ip_adapter (в т.ч. multi_image_mode: "mean" | "first").

    Returns:
        Тот же граф (in-place).
    """
    if "denoise_loop" not in graph.nodes:
        raise ValueError("Graph must have a 'denoise_loop' node")

    if "ip_adapter" in graph.nodes or "ip_image_encoder" in graph.nodes:
        return graph  # already added

    # Image encoder (CLIP vision) — для получения image_features
    encoder_pretrained = image_encoder_pretrained or "openai/clip-vit-large-patch14"
    image_encoder = BlockBuilder.build({
        "type": "conditioner/clip_vision",
        "pretrained": encoder_pretrained,
    })
    # Берём размерность выхода энкодера, чтобы IP-Adapter совпадал с любым CLIP (768 ViT-L, 1024 ViT-H)
    image_embed_dim = getattr(image_encoder, "embedding_dim", None) or (
        768 if "clip-vit-large" in (encoder_pretrained or "") else 1024
    )
    # Переопределение из kwargs имеет приоритет
    if "image_embed_dim" in ip_adapter_kwargs:
        image_embed_dim = ip_adapter_kwargs.pop("image_embed_dim")

    # cross_attention_dim: единственный источник — граф (metadata или обход backbone) §S3
    if cross_attention_dim is None:
        meta = getattr(graph, "metadata", None) or {}
        cross_attention_dim = meta.get("cross_attention_dim") or get_cross_attention_dim_from_graph(graph)
    if cross_attention_dim is None and "cross_attention_dim" not in ip_adapter_kwargs:
        raise ValueError(
            "IP-Adapter requires cross_attention_dim. Either build the backbone first (e.g. call graph.to(device) before adding IP-Adapter), "
            "or pass cross_attention_dim=... when adding the adapter (e.g. from your model's config)."
        )
    if "cross_attention_dim" in ip_adapter_kwargs:
        cross_attention_dim = ip_adapter_kwargs.pop("cross_attention_dim")

    # IP-Adapter: image_embed_dim = выход энкодера; cross_attention_dim по модели
    ip_config = {
        "type": "adapter/ip_adapter",
        "scale": ip_adapter_scale,
        "image_embed_dim": image_embed_dim,
        "cross_attention_dim": cross_attention_dim,
        **ip_adapter_kwargs,
    }
    ip_adapter = BlockBuilder.build(ip_config)

    graph.add_node("ip_image_encoder", image_encoder)
    graph.add_node("ip_adapter", ip_adapter)
    graph.connect("ip_image_encoder", "embedding", "ip_adapter", "image_features")

    if image_source is not None:
        src_node, src_port = image_source
        if src_node not in graph.nodes:
            raise ValueError(f"image_source node '{src_node}' not found in graph")
        graph.connect(src_node, src_port, "ip_image_encoder", "raw_condition")
    else:
        graph.expose_input("ip_image", "ip_image_encoder", "raw_condition")

    # IP-Adapter output can be wired to loop if backbone accepts ip_adapter_embeds
    # (optional: connect to denoise_loop.ip_adapter_embeds when loop supports it)
    return graph


def add_adapter_to_graph(
    graph: ComputeGraph,
    adapter_type: str,
    input_source: Optional[SourceRef] = None,
    **kwargs: Any,
) -> ComputeGraph:
    """Универсальное добавление адаптера к графу по типу.

    Args:
        graph: Целевой граф.
        adapter_type: "controlnet" | "ip_adapter" | ...
        input_source: Для controlnet — (node, port) для control_image.
                      Для ip_adapter — (node, port) для ip_image.
        **kwargs: Параметры адаптера (pretrained, scale, и т.д.).

    Returns:
        Тот же граф (in-place).
    """
    adapter_type = adapter_type.lower().strip()
    if adapter_type == "controlnet":
        return add_controlnet_to_graph(
            graph,
            control_image_source=input_source,
            **kwargs,
        )
    if adapter_type == "ip_adapter":
        return add_ip_adapter_to_graph(
            graph,
            image_source=input_source,
            **kwargs,
        )
    if adapter_type == "t2i_adapter" or adapter_type == "t2i":
        return add_t2i_adapter_to_graph(
            graph,
            control_image_source=input_source,
            **kwargs,
        )
    raise ValueError(
        f"Unknown adapter_type '{adapter_type}'. "
        "Supported: controlnet, ip_adapter, t2i_adapter."
    )


def add_t2i_adapter_to_graph(
    graph: ComputeGraph,
    t2i_pretrained: Optional[str] = None,
    adapter_type: str = "depth",
    scale: float = 1.0,
    fp16: bool = True,
    control_image_source: Optional[SourceRef] = None,
    **t2i_kwargs: Any,
) -> ComputeGraph:
    """Добавить T2I-Adapter к графу (аналогично ControlNet).

    Внутренний шаг должен содержать backbone с поддержкой adapter_features.
    При отсутствии поддержки (DiT, WAN и т.д.) граф не изменяется.
    """
    if "denoise_loop" not in graph.nodes:
        return graph
    loop = graph.nodes["denoise_loop"]
    if not hasattr(loop, "graph") or loop.graph is None:
        return graph
    inner = loop.graph
    if "t2i_adapter" in inner.nodes:
        return graph
    if "backbone" not in inner.nodes:
        return graph
    backbone = inner.nodes["backbone"]
    if not _backbone_supports_adapters(backbone):
        return graph

    config = {
        "type": "adapter/t2i",
        "adapter_type": adapter_type,
        "scale": scale,
        "fp16": fp16,
        **({"pretrained": t2i_pretrained} if t2i_pretrained else {}),
        **t2i_kwargs,
    }
    t2i = BlockBuilder.build(config)
    inner.add_node("t2i_adapter", t2i)
    inner.expose_input("t2i_control_image", "t2i_adapter", "control_image")
    inner.expose_input("condition", "t2i_adapter", "encoder_hidden_states")
    inner.expose_input("latents", "t2i_adapter", "sample")
    inner.expose_input("timestep", "t2i_adapter", "timestep")
    inner.connect("t2i_adapter", "output", "backbone", "adapter_features")

    if control_image_source is not None:
        src_node, src_port = control_image_source
        if src_node in graph.nodes:
            graph.connect(src_node, src_port, "denoise_loop", "t2i_control_image")
    else:
        graph.expose_input("t2i_control_image", "denoise_loop", "t2i_control_image")
    return graph


def add_optional_adapters_to_graph(
    graph: ComputeGraph,
    controlnet: bool = True,
    t2i_adapter: bool = True,
    ip_adapter: bool = True,
    controlnet_pretrained: str = "lllyasviel/control_v11p_sd15_canny",
    t2i_pretrained: Optional[str] = None,
    **kwargs: Any,
) -> ComputeGraph:
    """Добавить все применимые адаптеры к графу (video/audio/image с denoise_loop).

    Добавляет ControlNet и T2I-Adapter только если внутренний backbone поддерживает
    adapter_features (UNet2D, UNet3D). IP-Adapter добавляет энкодер и вход ip_image;
    для полной работы требуется инъекция в backbone (inject_into).
    """
    if controlnet:
        add_controlnet_to_graph(
            graph,
            controlnet_pretrained=controlnet_pretrained,
            **{k: v for k, v in kwargs.items() if k in ("conditioning_scale", "fp16")},
        )
    if t2i_adapter:
        add_t2i_adapter_to_graph(
            graph,
            t2i_pretrained=t2i_pretrained,
            **{k: v for k, v in kwargs.items() if k in ("adapter_type", "scale", "fp16")},
        )
    if ip_adapter:
        add_ip_adapter_to_graph(
            graph,
            **{k: v for k, v in kwargs.items() if k in ("ip_adapter_scale", "image_encoder_pretrained")},
        )
    return graph
