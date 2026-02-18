# yggdrasil/core/graph/adapters.py
"""Подключение адаптеров (ControlNet, IP-Adapter и др.) к любому графу.

Любой адаптер можно добавить к графу с denoise_loop и backbone, поддерживающим
adapter_features. Исходные данные для адаптера можно передавать как новый вход
графа или по ссылке на выход другого узла (node_name, port_name).
"""
from __future__ import annotations

import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union

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


def set_ip_adapter_scale_on_unet(
    unet: Any,
    scale: Union[float, List[float], Dict[str, Any]],
) -> None:
    """Set IP-Adapter strength on attention processors (Diffusers-style).

    scale:
        - float: same strength for all layers and images.
        - List[float]: per-image scales (e.g. [0.7, 0.8] for 2 images).
        - Dict (InstantStyle): per-block scale. Keys are attn name prefixes; values are
          scale or list. E.g. {"down_blocks.2": [0.0, 1.0], "up_blocks.0": [0.0, 1.0, 0.0]}.
          Layers not in dict get 0 (disabled).
    """
    from diffusers.models.attention_processor import IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0

    if not getattr(unet, "attn_processors", None):
        return
    # Flatten nested dict {"down": {"block_2": s}, "up": {"block_0": s}} -> {"down_blocks.2": s, "up_blocks.0": s}
    scale_dict: Optional[Dict[str, Any]] = None
    scale_list: Optional[List[float]] = None
    if isinstance(scale, dict):
        flat: Dict[str, Any] = {}
        for k, v in scale.items():
            if isinstance(v, dict):
                dir_prefix = "down_blocks." if k == "down" else "up_blocks." if k == "up" else f"{k}."
                for bk, bv in v.items():
                    block_id = bk.replace("block_", "")
                    flat[f"{dir_prefix}{block_id}"] = bv
            else:
                flat[k] = v
        scale_dict = flat if flat else None
    elif isinstance(scale, (list, tuple)):
        scale_list = [float(s) for s in scale]
    else:
        scale_list = [float(scale)]

    for attn_name, proc in unet.attn_processors.items():
        if not isinstance(proc, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)):
            continue
        if scale_dict is not None:
            best_key, best_val = None, 0.0
            for k, v in scale_dict.items():
                if attn_name.startswith(k) and (best_key is None or len(k) > len(best_key or "")):
                    best_key, best_val = k, v
            val = best_val if best_key is not None else 0.0
            if isinstance(val, (list, tuple)):
                proc.scale = [float(x) for x in val]
            else:
                n = len(proc.scale) if isinstance(proc.scale, (list, tuple)) else 1
                proc.scale = [float(val)] * n
        else:
            n = len(proc.scale) if isinstance(proc.scale, (list, tuple)) else 1
            proc.scale = [scale_list[i % len(scale_list)] for i in range(n)]


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
    inner.expose_input("timestep", node_name, "timestep")
    # SDXL/Euler: ControlNet must receive scaled latents (parity with diffusers pipeline_controlnet_sd_xl)
    if "scale_input" in inner.nodes and hasattr(inner.nodes["scale_input"], "block_type"):
        inner.connect("scale_input", "scaled", node_name, "sample")
    else:
        inner.expose_input("latents", node_name, "sample")
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
    graph.connect("ip_image_encoder", "scales", "ip_adapter", "image_scales")

    if image_source is not None:
        src_node, src_port = image_source
        if src_node not in graph.nodes:
            raise ValueError(f"image_source node '{src_node}' not found in graph")
        graph.connect(src_node, src_port, "ip_image_encoder", "raw_condition")
    else:
        graph.expose_input("ip_image", "ip_image_encoder", "raw_condition")
    # Optional: pre-computed embeds (Diffusers ip_adapter_image_embeds). Bypasses encoder when provided.
    graph.expose_input("ip_image_embeds", "ip_adapter", "image_embeds")

    # Wire adapter output to denoise loop
    graph.connect("ip_adapter", "image_prompt_embeds", "denoise_loop", "image_prompt_embeds")

    # Set IP-Adapter attention processors on UNet
    loop_block = graph.nodes.get("denoise_loop")
    if loop_block is not None:
        inner = getattr(loop_block, "graph", None) or getattr(getattr(loop_block, "_loop", None), "graph", None)
        if inner and "backbone" in getattr(inner, "nodes", {}):
            backbone = inner.nodes["backbone"]
            unet = getattr(backbone, "unet", None)
            if unet is not None and hasattr(unet, "set_attn_processor"):
                backbone._ip_adapter_scale = ip_adapter_scale
                attn_procs = getattr(unet, "attn_processors", None)
                if attn_procs and len(attn_procs) > 0 and getattr(backbone, "_ip_adapter_original_processors", None) is None:
                    backbone._ip_adapter_original_processors = dict(attn_procs)
                set_ip_adapter_processors_on_unet(unet, scale=ip_adapter_scale)

    # Optional: spatial masks for region-specific IP conditioning (conditioner/ip_adapter_mask output)
    graph.expose_input("ip_adapter_masks", "denoise_loop", "ip_adapter_masks")

    return graph


def add_ip_adapter_plus_to_graph(
    graph: ComputeGraph,
    ip_adapter_scale: float = 0.6,
    image_source: Optional[SourceRef] = None,
    image_encoder_pretrained: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    cross_attention_dim: Optional[int] = None,
    **ip_adapter_kwargs: Any,
) -> ComputeGraph:
    """Add IP-Adapter Plus (patch-level, ViT-H) to graph.

    Uses CLIP ViT-H with output_mode='patches' and adapter/ip_adapter_plus.
    If base IP-Adapter already exists, adds merge block and wires both to loop.
    """
    if "denoise_loop" not in graph.nodes:
        raise ValueError("Graph must have a 'denoise_loop' node")
    if cross_attention_dim is None:
        meta = getattr(graph, "metadata", None) or {}
        cross_attention_dim = meta.get("cross_attention_dim") or get_cross_attention_dim_from_graph(graph)
    if cross_attention_dim is None:
        raise ValueError("IP-Adapter Plus requires cross_attention_dim")

    encoder_cfg = {
        "type": "conditioner/clip_vision",
        "pretrained": image_encoder_pretrained,
        "output_mode": "patches",
    }
    image_encoder = BlockBuilder.build(encoder_cfg)
    image_embed_dim = getattr(image_encoder, "embedding_dim", None) or 1280
    ip_config = {
        "type": "adapter/ip_adapter_plus",
        "scale": ip_adapter_scale,
        "image_embed_dim": image_embed_dim,
        "cross_attention_dim": cross_attention_dim,
        **ip_adapter_kwargs,
    }
    ip_adapter = BlockBuilder.build(ip_config)
    enc_name, adp_name = "ip_image_encoder_plus", "ip_adapter_plus"
    graph.add_node(enc_name, image_encoder)
    graph.add_node(adp_name, ip_adapter)
    graph.connect(enc_name, "patches", adp_name, "image_features")
    if image_source is not None:
        src_node, src_port = image_source
        if src_node in graph.nodes:
            graph.connect(src_node, src_port, enc_name, "raw_condition")
    else:
        graph.expose_input("ip_image_plus", enc_name, "raw_condition")

    denoise_loop_node = "denoise_loop"
    if "ip_adapter" in graph.nodes and "ip_adapter_merge" not in graph.nodes:
        graph.edges = [e for e in graph.edges if not (
            e.src_node == "ip_adapter" and e.src_port == "image_prompt_embeds"
            and e.dst_node == denoise_loop_node and e.dst_port == "image_prompt_embeds"
        )]
        merge = BlockBuilder.build({"type": "adapter/ip_adapter_merge"})
        graph.add_node("ip_adapter_merge", merge)
        graph.connect("ip_adapter", "image_prompt_embeds", "ip_adapter_merge", "embeds_0")
        graph.connect(adp_name, "image_prompt_embeds", "ip_adapter_merge", "embeds_1")
        graph.connect("ip_adapter_merge", "image_prompt_embeds", denoise_loop_node, "image_prompt_embeds")
    elif "ip_adapter_merge" in graph.nodes:
        graph.connect(adp_name, "image_prompt_embeds", "ip_adapter_merge", "embeds_1")
    else:
        graph.connect(adp_name, "image_prompt_embeds", denoise_loop_node, "image_prompt_embeds")

    loop_block = graph.nodes.get(denoise_loop_node)
    if loop_block is not None:
        inner = getattr(loop_block, "graph", None) or getattr(getattr(loop_block, "_loop", None), "graph", None)
        if inner and "backbone" in getattr(inner, "nodes", {}):
            backbone = inner.nodes["backbone"]
            unet = getattr(backbone, "unet", None)
            if unet is not None and hasattr(unet, "set_attn_processor"):
                backbone._ip_adapter_scale = ip_adapter_scale
                attn_procs = getattr(unet, "attn_processors", None)
                if attn_procs and len(attn_procs) > 0 and not getattr(backbone, "_ip_adapter_original_processors", None):
                    backbone._ip_adapter_original_processors = dict(attn_procs)
                set_ip_adapter_processors_on_unet(unet, scale=ip_adapter_scale, num_tokens=(16,))
    graph.expose_input("ip_adapter_masks", denoise_loop_node, "ip_adapter_masks")
    return graph


def add_ip_adapter_faceid_to_graph(
    graph: ComputeGraph,
    ip_adapter_scale: float = 0.6,
    image_source: Optional[SourceRef] = None,
    cross_attention_dim: Optional[int] = None,
    **ip_adapter_kwargs: Any,
) -> ComputeGraph:
    """Add IP-Adapter FaceID to graph.

    Uses conditioner/faceid (InsightFace) and adapter/ip_adapter_faceid.
    Requires: pip install insightface
    """
    if "denoise_loop" not in graph.nodes:
        raise ValueError("Graph must have a 'denoise_loop' node")
    if cross_attention_dim is None:
        meta = getattr(graph, "metadata", None) or {}
        cross_attention_dim = meta.get("cross_attention_dim") or get_cross_attention_dim_from_graph(graph)
    if cross_attention_dim is None:
        raise ValueError("IP-Adapter FaceID requires cross_attention_dim")

    face_encoder = BlockBuilder.build({"type": "conditioner/faceid"})
    ip_config = {
        "type": "adapter/ip_adapter_faceid",
        "scale": ip_adapter_scale,
        "face_embed_dim": 512,
        "cross_attention_dim": cross_attention_dim,
        "num_tokens": 4,
        **ip_adapter_kwargs,
    }
    ip_adapter = BlockBuilder.build(ip_config)
    enc_name, adp_name = "ip_faceid_encoder", "ip_adapter_faceid"
    graph.add_node(enc_name, face_encoder)
    graph.add_node(adp_name, ip_adapter)
    graph.connect(enc_name, "embedding", adp_name, "image_features")
    if image_source is not None:
        src_node, src_port = image_source
        if src_node in graph.nodes:
            graph.connect(src_node, src_port, enc_name, "raw_condition")
    else:
        graph.expose_input("ip_face_image", enc_name, "raw_condition")

    denoise_loop_node = "denoise_loop"
    if "ip_adapter" in graph.nodes or "ip_adapter_plus" in graph.nodes:
        if "ip_adapter_merge" not in graph.nodes:
            merge = BlockBuilder.build({"type": "adapter/ip_adapter_merge"})
            graph.add_node("ip_adapter_merge", merge)
            prev = "ip_adapter" if "ip_adapter" in graph.nodes else "ip_adapter_plus"
            graph.edges = [e for e in graph.edges if not (
                e.src_node == prev and e.src_port == "image_prompt_embeds"
                and e.dst_node == denoise_loop_node and e.dst_port == "image_prompt_embeds"
            )]
            graph.connect(prev, "image_prompt_embeds", "ip_adapter_merge", "embeds_0")
            graph.connect(adp_name, "image_prompt_embeds", "ip_adapter_merge", "embeds_1")
            graph.connect("ip_adapter_merge", "image_prompt_embeds", denoise_loop_node, "image_prompt_embeds")
        else:
            graph.connect(adp_name, "image_prompt_embeds", "ip_adapter_merge", "embeds_2")
    else:
        graph.connect(adp_name, "image_prompt_embeds", denoise_loop_node, "image_prompt_embeds")

    loop_block = graph.nodes.get(denoise_loop_node)
    if loop_block is not None:
        inner = getattr(loop_block, "graph", None) or getattr(getattr(loop_block, "_loop", None), "graph", None)
        if inner and "backbone" in getattr(inner, "nodes", {}):
            backbone = inner.nodes["backbone"]
            unet = getattr(backbone, "unet", None)
            if unet is not None and hasattr(unet, "set_attn_processor"):
                backbone._ip_adapter_scale = ip_adapter_scale
                attn_procs = getattr(unet, "attn_processors", None)
                if attn_procs and len(attn_procs) > 0 and not getattr(backbone, "_ip_adapter_original_processors", None):
                    backbone._ip_adapter_original_processors = dict(attn_procs)
                set_ip_adapter_processors_on_unet(unet, scale=ip_adapter_scale)
    graph.expose_input("ip_adapter_masks", denoise_loop_node, "ip_adapter_masks")
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
    if adapter_type == "ip_adapter_plus":
        return add_ip_adapter_plus_to_graph(graph, image_source=input_source, **kwargs)
    if adapter_type == "ip_adapter_faceid":
        return add_ip_adapter_faceid_to_graph(graph, image_source=input_source, **kwargs)
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
