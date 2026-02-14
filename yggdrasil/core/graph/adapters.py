# yggdrasil/core/graph/adapters.py
"""Подключение адаптеров (ControlNet, IP-Adapter и др.) к любому графу.

Любой адаптер можно добавить к графу с denoise_loop и backbone, поддерживающим
adapter_features. Исходные данные для адаптера можно передавать как новый вход
графа или по ссылке на выход другого узла (node_name, port_name).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

from .graph import ComputeGraph
from ..block.builder import BlockBuilder


# Тип "источник по ссылке": (имя_узла, имя_порта)
SourceRef = Tuple[str, str]


def add_controlnet_to_graph(
    graph: ComputeGraph,
    controlnet_pretrained: str = "lllyasviel/control_v11p_sd15_canny",
    conditioning_scale: float = 1.0,
    fp16: bool = True,
    control_image_source: Optional[SourceRef] = None,
    **controlnet_kwargs: Any,
) -> ComputeGraph:
    """Добавить ControlNet как отдельный блок к существующему графу.

    Граф должен содержать узел ``denoise_loop`` (LoopSubGraph). Внутренний
    шаг должен содержать backbone с поддержкой ``adapter_features``
    (например ``backbone/unet2d_condition``). Подходит для SD 1.5 и SDXL
    (для SDXL укажите controlnet_pretrained для SDXL, напр. xinsir/controlnet-sdxl-1.0-canny).

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
    if "denoise_loop" not in graph.nodes:
        raise ValueError("Graph must have a 'denoise_loop' node (LoopSubGraph)")

    loop = graph.nodes["denoise_loop"]
    if not hasattr(loop, "graph") or loop.graph is None:
        raise ValueError("denoise_loop must be a LoopSubGraph with an inner graph")

    inner = loop.graph
    if "controlnet" in inner.nodes:
        return graph  # already has ControlNet

    if "backbone" not in inner.nodes:
        raise ValueError("Inner graph must have a 'backbone' node")

    backbone = inner.nodes["backbone"]
    bt = getattr(backbone, "block_type", "")
    # Batched backbone now supports adapter_features (ControlNet residuals)

    config = {
        "type": "adapter/controlnet",
        "pretrained": controlnet_pretrained,
        "conditioning_scale": conditioning_scale,
        "fp16": fp16,
        **controlnet_kwargs,
    }
    controlnet = BlockBuilder.build(config)
    inner.add_node("controlnet", controlnet)

    inner.expose_input("control_image", "controlnet", "control_image")
    inner.expose_input("condition", "controlnet", "encoder_hidden_states")
    inner.expose_input("latents", "controlnet", "sample")
    inner.expose_input("timestep", "controlnet", "timestep")
    inner.connect("controlnet", "output", "backbone", "adapter_features")

    if control_image_source is not None:
        src_node, src_port = control_image_source
        if src_node not in graph.nodes:
            raise ValueError(f"control_image_source node '{src_node}' not found in graph")
        graph.connect(src_node, src_port, "denoise_loop", "control_image")
    else:
        graph.expose_input("control_image", "denoise_loop", "control_image")
    return graph


def add_ip_adapter_to_graph(
    graph: ComputeGraph,
    ip_adapter_scale: float = 0.6,
    image_source: Optional[SourceRef] = None,
    image_encoder_pretrained: Optional[str] = None,
    **ip_adapter_kwargs: Any,
) -> ComputeGraph:
    """Добавить IP-Adapter к графу: блок кодирования изображения + IP-Adapter.

    Добавляет узел image_encoder (CLIP vision) и adapter/ip_adapter. Входное
    изображение можно передать новым входом графа ``ip_image`` или по ссылке
    (image_source=(node_name, port_name)). Для полной работы IP-Adapter backbone
    должен поддерживать инъекцию (inject_into) или порт ip_adapter_embeds.

    Args:
        graph: Граф с denoise_loop и backbone.
        ip_adapter_scale: Сила влияния image prompt.
        image_source: Если задано (node_name, port_name) — изображение берётся
            с выхода узла. Иначе добавляется вход графа ip_image.
        image_encoder_pretrained: HF model ID для CLIP image encoder (по умолчанию из контекста).
        **ip_adapter_kwargs: Доп. параметры для adapter/ip_adapter.

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

    # IP-Adapter: cross_attention_dim 768 для SD 1.5, 2048 для SDXL (concat text)
    ip_config = {
        "type": "adapter/ip_adapter",
        "scale": ip_adapter_scale,
        "image_embed_dim": 1024,
        "cross_attention_dim": 768,
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
    raise ValueError(
        f"Unknown adapter_type '{adapter_type}'. "
        "Supported: controlnet, ip_adapter."
    )
