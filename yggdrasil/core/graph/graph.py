# yggdrasil/core/graph/graph.py
"""ComputeGraph — направленный ациклический граф из блоков.

Это главная структура данных YggDrasil v2.
Пайплайн = граф из блоков, соединённых через порты.
"""
from __future__ import annotations

import copy
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Edge — ребро графа
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Edge:
    """Соединение между выходным портом одного узла и входным портом другого.
    
    Attributes:
        src_node: Имя узла-источника.
        src_port: Имя выходного порта источника.
        dst_node: Имя узла-приёмника.
        dst_port: Имя входного порта приёмника.
    """
    src_node: str
    src_port: str
    dst_node: str
    dst_port: str

    def __repr__(self) -> str:
        return f"{self.src_node}.{self.src_port} -> {self.dst_node}.{self.dst_port}"


class _DeferredPlaceholder:
    """Placeholder for a node whose block is built later (in parallel) in to(device)."""
    __slots__ = ("block_type",)

    def __init__(self, block_type: str):
        self.block_type = block_type


# ---------------------------------------------------------------------------
# ComputeGraph — DAG
# ---------------------------------------------------------------------------

class ComputeGraph:
    """Направленный ациклический граф блоков — настоящий Lego-конструктор.
    
    Позволяет:
    - Добавлять/удалять/заменять узлы
    - Соединять порты блоков рёбрами
    - Объявлять входы/выходы графа
    - Валидировать корректность
    - Выполнять граф (через GraphExecutor)
    - Сериализовать в YAML и загружать обратно
    - Визуализировать как Mermaid-диаграмму
    
    Пример::
    
        graph = ComputeGraph("sd15_txt2img")
        graph.add_node("clip", clip_block)
        graph.add_node("unet", unet_block)
        graph.connect("clip", "embedding", "unet", "condition")
        graph.expose_input("prompt", "clip", "text")
        graph.expose_output("noise_pred", "unet", "output")
    """
    
    def __init__(self, name: str = "unnamed"):
        self.name: str = name
        self.nodes: OrderedDict[str, Any] = OrderedDict()  # name -> AbstractBaseBlock
        self.edges: List[Edge] = []
        # Fan-out: one graph input can feed multiple (node, port) targets
        self.graph_inputs: Dict[str, List[Tuple[str, str]]] = {}   # input_name -> [(node, port), ...]
        self.graph_outputs: Dict[str, Tuple[str, str]] = {}        # output_name -> (node, port)
        self.metadata: Dict[str, Any] = {}
        # Device tracking
        self._device: Any = None
        self._dtype: Any = None
        # Deferred building: (node_name, cfg, target_inner). target_inner None = top-level; str = add to that node's inner graph.
        self._deferred: List[Tuple[str, Dict[str, Any], Optional[str]]] = []
        # Orchestrator: single point of control for build phases (ref REFACTORING_GRAPH_PIPELINE_ENGINE.md)
        self._orchestrator: Optional[Any] = None

    def _get_orchestrator(self) -> Optional[Any]:
        """Lazy init orchestrator; sync metadata from graph."""
        if self._orchestrator is None:
            try:
                from .orchestrator import GraphBuildOrchestrator
                self._orchestrator = GraphBuildOrchestrator(self)
                self._orchestrator._state.metadata = dict(self.metadata)
            except Exception:
                return None
        else:
            self._orchestrator._state.metadata = dict(self.metadata)
        return self._orchestrator

    # ==================== DEVICE MANAGEMENT ====================
    
    def to(self, device=None, dtype=None) -> ComputeGraph:
        """Перенести весь граф на устройство.
        
        Рекурсивно переносит все узлы, включая вложенные SubGraph.
        Если есть отложенные узлы (UNet, VAE), сначала собирает их параллельно.
        
        Когда dtype=None (по умолчанию), каждый блок сохраняет свой
        оригинальный dtype. Это позволяет UNet оставаться в float16
        на MPS (быстрее и экономичнее), а CLIP — в float32
        (нужен для LayerNorm).
        
        Args:
            device: Устройство ("cuda", "mps", "cpu", torch.device).
            dtype: Тип данных. Если None — блоки сохраняют свой dtype.
                   Если указан — все блоки конвертируются.
        
        Returns:
            self (для chaining).
        """
        import os
        import torch

        if isinstance(device, str):
            device = torch.device(device)

        self._device = device
        self._dtype = dtype  # None means "keep original per-block dtype"

        # На CUDA уменьшаем фрагментацию (см. PyTorch Memory Management)
        if device.type == "cuda":
            if "PYTORCH_ALLOC_CONF" not in os.environ:
                os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

        def _maybe_empty_cache():
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Build deferred (heavy) blocks in parallel before moving to device
        if self._deferred:
            orch = self._get_orchestrator()
            if orch is not None:
                orch.materialize()
            else:
                self._materialize_deferred()

        # Переносим узлы по одному и после каждого сбрасываем кэш CUDA — меньше шанс OOM из‑за фрагментации
        for name, block in self.nodes.items():
            self._move_block(block, device, dtype)
            _maybe_empty_cache()
            # Recurse into SubGraphs (LoopSubGraph, SubGraph)
            if hasattr(block, 'graph') and block.graph is not None:
                block.graph.to(device, dtype)
                _maybe_empty_cache()
            # Recurse into wrapper blocks that hold _loop (e.g. DenoiseLoopSDXLBlock)
            if hasattr(block, '_loop') and block._loop is not None:
                self._move_block(block._loop, device, dtype)
                _maybe_empty_cache()
                if hasattr(block._loop, 'graph') and getattr(block._loop, 'graph', None) is not None:
                    block._loop.graph.to(device, dtype)
                    _maybe_empty_cache()

        # IP-Adapter: процессоры могли не перенестись при материализации (UNet был на meta)
        self._ensure_ip_adapter_processors_on_device()
        return self

    def infer_metadata_if_needed(self) -> None:
        """Обновить метаданные из состава графа. Вызывается после add_node и replace_node.
        Подставляет только общие поля (default_num_steps, default_guidance_scale, modality,
        backbone_pretrained); не привязывается к конкретным моделям (SDXL/SD15 и т.д.).
        Существующие ключи в self.metadata не перезаписываются.
        """
        meta = self.metadata
        if meta is None:
            self.metadata = {}
            meta = self.metadata
        # backbone_pretrained — из узлов (loop/backbone) или из отложенных конфигов (для любых диффузий)
        pretrained = None
        for _n, block in self.nodes.items():
            bt = getattr(block, "block_type", "")
            if bt.startswith("loop/"):
                pretrained = getattr(block, "pretrained", None) or (
                    getattr(block, "_loop", None) and getattr(block._loop, "pretrained", None)
                )
                break
            if bt.startswith("backbone/"):
                pretrained = getattr(block, "pretrained", None)
                break
        for _name, cfg, _target in getattr(self, "_deferred", []):
            if isinstance(cfg, dict) and (cfg.get("type") or "").startswith("loop/"):
                pretrained = pretrained or cfg.get("pretrained")
                break
        if pretrained and "backbone_pretrained" not in meta:
            meta.setdefault("backbone_pretrained", pretrained)
        # base_model — из pretrained (нужно для IP-Adapter cross_attention_dim: sdxl=2048, sd15=768)
        if "base_model" not in meta and pretrained:
            p = (pretrained or "").lower()
            if "xl" in p or p.rstrip("/").endswith("xl"):
                meta.setdefault("base_model", "sdxl")
            else:
                meta.setdefault("base_model", "sd15")
        meta.setdefault("default_num_steps", 50)
        meta.setdefault("default_guidance_scale", 7.5)
        # default_width/height — автоматически из pretrained
        if "default_width" not in meta and pretrained:
            p = (pretrained or "").lower()
            if "xl" in p or p.rstrip("/").endswith("xl"):
                meta.setdefault("default_width", 1024)
                meta.setdefault("default_height", 1024)
            else:
                meta.setdefault("default_width", 512)
                meta.setdefault("default_height", 512)
        # modality — по наличию loop/codec
        if "modality" not in meta:
            has_loop = any(getattr(b, "block_type", "").startswith("loop/") for b in self.nodes.values())
            has_codec = any(getattr(b, "block_type", "").startswith("codec/") for b in self.nodes.values())
            if has_loop or has_codec:
                for _name, cfg, _ in getattr(self, "_deferred", []):
                    if isinstance(cfg, dict):
                        t = cfg.get("type") or cfg.get("block_type") or ""
                        if t.startswith("loop/"):
                            has_loop = True
                        if t.startswith("codec/"):
                            has_codec = True
                if has_loop and has_codec:
                    meta.setdefault("modality", "image")
        # use_euler_init_sigma — обязателен для EulerDiscreteScheduler (SDXL parity): начальный шум масштабируется init_noise_sigma
        if "use_euler_init_sigma" not in meta:
            for _n, block in self.nodes.items():
                inner = getattr(block, "graph", None) or (getattr(block, "_loop", None) and getattr(block._loop, "graph", None))
                if inner and getattr(inner, "nodes", None):
                    for _sn, sblock in inner.nodes.items():
                        if getattr(sblock, "block_type", "") == "solver/euler_discrete":
                            meta.setdefault("use_euler_init_sigma", True)
                            break
                if meta.get("use_euler_init_sigma"):
                    break
            # Отложенная сборка: loop/denoise_sdxl всегда использует Euler внутри
            if not meta.get("use_euler_init_sigma"):
                for _name, cfg, _ in getattr(self, "_deferred", []):
                    if isinstance(cfg, dict) and (cfg.get("type") or "").startswith("loop/"):
                        meta.setdefault("use_euler_init_sigma", True)
                        break
        # L4: latent_channels, spatial_scale_factor, prediction_type — единственный источник из графа для адаптеров и шаблонов
        if "latent_channels" not in meta or "spatial_scale_factor" not in meta:
            for _n, block in self.nodes.items():
                bt = getattr(block, "block_type", "")
                if bt.startswith("codec/"):
                    lc = getattr(block, "latent_channels", None)
                    if lc is not None and "latent_channels" not in meta:
                        meta.setdefault("latent_channels", int(lc))
                    sf = getattr(block, "spatial_scale_factor", None) or getattr(block, "scale_factor", None)
                    if sf is not None and "spatial_scale_factor" not in meta:
                        meta.setdefault("spatial_scale_factor", int(sf))
                    break
                if bt.startswith("loop/"):
                    inner = getattr(block, "graph", None) or (getattr(block, "_loop", None) and getattr(block._loop, "graph", None))
                    if inner and getattr(inner, "nodes", None):
                        for _sn, sblock in inner.nodes.items():
                            if getattr(sblock, "block_type", "").startswith("codec/"):
                                lc = getattr(sblock, "latent_channels", None)
                                if lc is not None and "latent_channels" not in meta:
                                    meta.setdefault("latent_channels", int(lc))
                                sf = getattr(sblock, "spatial_scale_factor", None) or getattr(sblock, "scale_factor", None)
                                if sf is not None and "spatial_scale_factor" not in meta:
                                    meta.setdefault("spatial_scale_factor", int(sf))
                                break
                    break
        if "prediction_type" not in meta:
            for _n, block in self.nodes.items():
                pred = getattr(block, "prediction_type", None)
                if pred is not None:
                    meta.setdefault("prediction_type", str(pred))
                    break
                if getattr(block, "block_type", "").startswith("loop/"):
                    inner = getattr(block, "graph", None) or (getattr(block, "_loop", None) and getattr(block._loop, "graph", None))
                    if inner and getattr(inner, "nodes", None):
                        for _sn, sblock in inner.nodes.items():
                            pred = getattr(sblock, "prediction_type", None)
                            if pred is not None:
                                meta.setdefault("prediction_type", str(pred))
                                break
                    break
        # A2: маппинг control_type → graph_input_name для нескольких ControlNet (единственный источник для пайплайна/UI)
        try:
            from .adapters import get_controlnet_input_mapping
            mapping = get_controlnet_input_mapping(self)
            if mapping:
                meta["controlnet_input_mapping"] = mapping
        except Exception:
            pass
        # S3: cross_attention_dim из графа — единственный источник для IP-Adapter и др. (без эвристик по имени)
        if "cross_attention_dim" not in meta:
            try:
                from .adapters import get_cross_attention_dim_from_graph
                dim = get_cross_attention_dim_from_graph(self)
                if dim is not None:
                    meta["cross_attention_dim"] = dim
            except Exception:
                pass

    def _materialize_deferred(self) -> None:
        """Build all deferred blocks in parallel (UNet, VAE, IP-Adapter, ControlNet), then replace/wire."""
        if not self._deferred:
            return
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from yggdrasil.core.block.builder import BlockBuilder

        # Normalize to 3-tuple (node_name, cfg, target_inner)
        items = []
        for item in self._deferred:
            if len(item) == 2:
                items.append((item[0], item[1], None))
            else:
                items.append(item)

        top_level = [(n, c) for n, c, t in items if t is None]
        adapters = [(n, c, t) for n, c, t in items if t is not None]
        # IP-Adapter must be built after encoder so image_embed_dim matches encoder output (e.g. 768 for ViT-L, 1024 for ViT-H)
        ip_adapter_top = [(n, c) for n, c in top_level if (c.get("type") or c.get("block_type")) == "adapter/ip_adapter"]
        other_top = [(n, c) for n, c in top_level if (c.get("type") or c.get("block_type")) != "adapter/ip_adapter"]

        def build_one(node_name: str, cfg: Dict) -> Any:
            block_type = (cfg.get("type") or cfg.get("block_type") or "")
            if block_type.startswith("loop/"):
                print("  Loading denoise loop (UNet)...", flush=True)
            elif block_type.startswith("codec/"):
                print("  Loading VAE (codec)...", flush=True)
            elif block_type == "adapter/ip_adapter":
                print("  Loading IP-Adapter...", flush=True)
            elif block_type == "adapter/controlnet":
                print("  Loading ControlNet...", flush=True)
            return BlockBuilder.build(cfg)

        print("Loading pipeline weights in parallel (one-time)...", flush=True)
        max_workers = min(4, len(other_top) + len(adapters) + (1 if ip_adapter_top else 0))
        # L1: pass graph metadata when building loop so step template can be chosen by solver_type/modality
        meta = getattr(self, "metadata", None) or {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # Phase 1: build top-level except IP-Adapter (loop, codec)
            built_top = {}
            if other_top:
                def _cfg_for_build(node_name: str, cfg: dict) -> dict:
                    c = dict(cfg)
                    if (c.get("type") or c.get("block_type") or "").startswith("loop/"):
                        c["_graph_metadata"] = dict(meta)
                    return c
                futures = {pool.submit(build_one, n, _cfg_for_build(n, c)): n for n, c in other_top}
                for fut in as_completed(futures):
                    node_name = futures[fut]
                    built_top[node_name] = fut.result()
            for node_name, block in built_top.items():
                self.nodes[node_name] = block

            # IP-Adapter: build encoder first, then adapter; cross_attention_dim — из уже собранного backbone (универсально)
            from .adapters import get_cross_attention_dim_from_graph
            for node_name, cfg in ip_adapter_top:
                encoder_pretrained = cfg.get("image_encoder_pretrained") or cfg.get("pretrained") or "openai/clip-vit-large-patch14"
                print("  Loading IP-Adapter image encoder (CLIP vision)...", flush=True)
                encoder = BlockBuilder.build({"type": "conditioner/clip_vision", "pretrained": encoder_pretrained})
                image_embed_dim = getattr(encoder, "embedding_dim", None) or (
                    768 if "clip-vit-large" in str(encoder_pretrained) else 1024
                )
                cross_attention_dim = (
                    cfg.get("cross_attention_dim")
                    or (getattr(self, "metadata", None) or {}).get("cross_attention_dim")
                    or get_cross_attention_dim_from_graph(self)
                )
                if cross_attention_dim is None:
                    raise ValueError(
                        "IP-Adapter: could not infer cross_attention_dim from graph backbone. "
                        "Ensure the denoise loop is built first, or set cross_attention_dim in the adapter config."
                    )
                ip_cfg = {k: v for k, v in cfg.items() if k not in ("image_encoder_pretrained", "pretrained")}
                ip_cfg["type"] = "adapter/ip_adapter"
                ip_cfg["image_embed_dim"] = image_embed_dim
                ip_cfg.setdefault("cross_attention_dim", cross_attention_dim)
                print("  Loading IP-Adapter...", flush=True)
                ip_adapter = BlockBuilder.build(ip_cfg)
                encoder_name = "ip_image_encoder"
                if encoder_name in self.nodes:
                    encoder_name = f"{encoder_name}_{len(self.nodes)}"
                self.nodes[encoder_name] = encoder
                self.nodes[node_name] = ip_adapter
                self.connect(encoder_name, "embedding", node_name, "image_features")
                self.expose_input("ip_image", encoder_name, "raw_condition")

                # Wire IP-Adapter output into denoise loop and inject processors into UNet
                denoise_loop_node = None
                for n, block in self.nodes.items():
                    if getattr(block, "block_type", "").startswith("loop/") or hasattr(block, "graph") or (hasattr(block, "_loop") and getattr(block._loop, "graph", None)):
                        denoise_loop_node = n
                        break
                if denoise_loop_node:
                    self.connect(node_name, "image_prompt_embeds", denoise_loop_node, "image_prompt_embeds")
                    inner = getattr(self.nodes[denoise_loop_node], "graph", None) or (getattr(self.nodes[denoise_loop_node], "_loop", None) and getattr(self.nodes[denoise_loop_node]._loop, "graph", None))
                    if inner:
                        inodes = getattr(inner, "nodes", {}) or {}
                        backbone = inodes.get("backbone")
                        if backbone is None:
                            for _n, b in inodes.items():
                                if getattr(b, "block_type", "").startswith("backbone/"):
                                    backbone = b
                                    break
                        if backbone is not None:
                            unet = getattr(backbone, "unet", None)
                            if unet is not None and hasattr(unet, "set_attn_processor"):
                                from .adapters import set_ip_adapter_processors_on_unet
                                scale = float(cfg.get("scale", 0.6))
                                backbone._ip_adapter_scale = scale
                                # Сохраняем исходные процессоры, чтобы при вызове без ip_image включать «голый» пайплайн
                                attn_procs = getattr(unet, "attn_processors", None)
                                if attn_procs is not None and len(attn_procs) > 0:
                                    backbone._ip_adapter_original_processors = dict(attn_procs)
                                    set_ip_adapter_processors_on_unet(unet, scale=scale)
                                    backbone._ip_adapter_processors = dict(unet.attn_processors)
                                else:
                                    # UNet на meta: attn_processors ещё пустой; отложить инициализацию до первого вызова с ip_image
                                    backbone._ip_adapter_original_processors = None
                                    backbone._ip_adapter_processors = None

            # Phase 2: build adapters (controlnet) and add to inner graph
            for node_name, cfg, target_inner in adapters:
                block = build_one(node_name, cfg)
                from .adapters import add_controlnet_to_graph
                add_controlnet_to_graph(
                    self,
                    controlnet_block=block,
                    denoise_loop_node=target_inner,
                    controlnet_node_name=node_name,
                )

        self._deferred.clear()

    @staticmethod
    def _move_block(block, device, dtype):
        """Move a single block to device/dtype."""
        import logging
        _logger = logging.getLogger(__name__)
        
        if not hasattr(block, 'to'):
            return
        try:
            if dtype is not None:
                block.to(device=device, dtype=dtype)
            else:
                block.to(device)
        except TypeError:
            # Some blocks don't accept dtype kwarg — retry with device only
            try:
                block.to(device)
            except Exception as e:
                _logger.warning(
                    f"Failed to move block {getattr(block, 'block_type', type(block).__name__)} "
                    f"to {device}: {e}"
                )
        except Exception as e:
            _logger.warning(
                f"Failed to move block {getattr(block, 'block_type', type(block).__name__)} "
                f"to {device}/{dtype}: {e}"
            )
    
    @property
    def device(self):
        """Текущее устройство графа."""
        return self._device
    
    # ==================== СБОРКА ГРАФА ====================
    
    def add_node(
        self,
        name: str | None = None,
        block: Any = None,
        *,
        type: str | None = None,
        auto_connect: bool = True,
        target_inner: str | None = None,
        **config,
    ) -> ComputeGraph:
        """Добавить узел в граф.

        Режимы:
        1. add_node(name, block) — добавить готовый блок.
        2. add_node(type="...", name="...", ...) — BlockBuilder собирает блок по type и config,
           при auto_connect=True подключает по правилам роли (role_rules).
        3. add_node(type="adapter/controlnet", target_inner="denoise_loop", ...) — добавить узел
           во внутренний граф указанного узла (например denoise_loop) и пробросить вход на верхний граф.

        Block builder вызывается внутри при передаче type=; снаружи разработчик видит только add_node.

        Args:
            name: Уникальное имя узла.
            block: Готовый блок (режим 1).
            type: block_type из реестра (режимы 2–3).
            auto_connect: If True, wire by role (adapters) and run pipeline auto-wire (conditioner -> denoise_loop -> codec, expose inputs/outputs). Set False for custom pipelines and use connect/expose_input/expose_output manually.
            target_inner: Опционально. Для ControlNet не задаётся — место подключения определяется автоматически.
            **config: Параметры для BlockBuilder (pretrained= для ControlNet и для IP-Adapter image encoder и т.д.).

        Returns:
            self (для chaining).
        """
        if block is not None:
            if name is None:
                raise ValueError("name required when passing block")
            if name in self.nodes:
                raise ValueError(f"Node '{name}' already exists in graph '{self.name}'")
            self.nodes[name] = block
            self.infer_metadata_if_needed()
            return self

        if type is None:
            raise ValueError("Either (name, block) or type=... required")
        if not isinstance(type, str):
            raise ValueError(
                "type= must be a string (e.g. type='conditioner/clip_sdxl', type='adapter/controlnet'). "
                "Do not use type=... (Ellipsis)."
            )

        from yggdrasil.core.block.builder import BlockBuilder
        from yggdrasil.core.graph.role_rules import (
            get_role_for_block_type,
            get_connection_rules,
            get_default_config_for_block_type,
            resolve_inner_target_for_adapter,
            resolve_loop_for_backbone,
        )

        # Адаптеры (ControlNet, T2I): target_inner не задан — подставляем узел цикла денойзинга автоматически
        if target_inner is None and type.startswith("adapter/"):
            target_inner = resolve_inner_target_for_adapter(self.nodes, type)

        meta = getattr(self, "metadata", None) or {}
        defaults = get_default_config_for_block_type(type, meta)
        config_filter = {"target_inner"}
        if type != "adapter/ip_adapter":
            config_filter.add("image_encoder_pretrained")
        cfg = {**defaults, **{k: v for k, v in config.items() if k not in config_filter}}
        cfg["type"] = type
        if type == "adapter/ip_adapter" and config.get("pretrained") and not cfg.get("image_encoder_pretrained"):
            cfg["image_encoder_pretrained"] = config["pretrained"]
        if type.startswith("loop/"):
            cfg["num_steps"] = int(meta.get("default_num_steps", 50))
            cfg["guidance_scale"] = float(meta.get("default_guidance_scale", 7.5))
        resolved_loop = False
        if target_inner is None and type.startswith("backbone/"):
            resolved = resolve_loop_for_backbone(type, cfg, meta)
            if resolved is not None:
                type, cfg = resolved
                resolved_loop = True
        # ControlNet: target_inner уже подставлен (или задан пользователем). Откладываем сборку до to(device).
        if type == "adapter/controlnet" and target_inner:
            cfg_plain = OmegaConf.to_container(cfg, resolve=True) if hasattr(cfg, "to_container") else dict(cfg)
            adapter_node_name = name or "controlnet"
            if adapter_node_name in self.nodes:
                adapter_node_name = f"{adapter_node_name}_{len(self.nodes)}"
            self._deferred.append((adapter_node_name, cfg_plain, target_inner))
            orch = self._get_orchestrator()
            if orch is not None:
                orch.register_node(adapter_node_name, type, cfg_plain, target_inner=target_inner)
                orch.invalidate()
            if auto_connect:
                from .pipeline_auto_wire import apply_pipeline_auto_wire
                apply_pipeline_auto_wire(self)
            self.infer_metadata_if_needed()
            return self
        ip_adapter_with_encoder = type == "adapter/ip_adapter" and (cfg.get("image_encoder_pretrained") or cfg.get("pretrained"))
        node_name = name or type.replace("/", "_").replace(".", "_")
        if node_name in self.nodes:
            node_name = f"{node_name}_{len(self.nodes)}"

        # Defer heavy blocks (loop, codec, IP-Adapter) to load in parallel in to(device)
        if not target_inner and (type.startswith("loop/") or type.startswith("codec/") or type == "adapter/ip_adapter"):
            if type == "adapter/ip_adapter" and not ip_adapter_with_encoder:
                pass  # no encoder config, build now
            else:
                cfg_plain = OmegaConf.to_container(cfg, resolve=True) if hasattr(cfg, "to_container") else dict(cfg)
                self._deferred.append((node_name, cfg_plain, None))
                self.nodes[node_name] = _DeferredPlaceholder(type)
                orch = self._get_orchestrator()
                if orch is not None:
                    orch.register_node(node_name, type, cfg_plain)
                    if type.startswith("loop/"):
                        orch.update_denoise_loop_name(node_name)
                    orch.invalidate()
                if auto_connect:
                    from .pipeline_auto_wire import apply_pipeline_auto_wire
                    apply_pipeline_auto_wire(self)
                self.infer_metadata_if_needed()
                return self

        if ip_adapter_with_encoder:
            block = None
        else:
            block = BlockBuilder.build(cfg)
            if resolved_loop and block is not None:
                bt = getattr(block, "block_type", "")
                if not (bt and bt.startswith("loop/")):
                    import logging
                    logging.getLogger(__name__).warning(
                        "Backbone was resolved to loop but built block has block_type=%r; ensure BlockBuilder builds loop for type=%r",
                        bt, cfg.get("type"),
                    )

        if target_inner:
            if target_inner not in self.nodes:
                raise ValueError(f"target_inner node '{target_inner}' not found")
            inner_block = self.nodes[target_inner]
            inner = getattr(inner_block, "graph", None) or (
                getattr(inner_block, "_loop", None) and getattr(inner_block._loop, "graph", None)
            )
            if inner is None:
                raise ValueError(f"'{target_inner}' has no inner graph (need .graph or ._loop.graph)")
            if node_name in inner.nodes:
                node_name = f"{node_name}_{len(inner.nodes)}"
            inner.nodes[node_name] = block
            if auto_connect:
                role = get_role_for_block_type(type)
                rules = get_connection_rules(role)
                if rules:
                    target_node = rules.get("target_node")
                    target_port = rules.get("target_port")
                    graph_input = rules.get("graph_input")
                    out_port = rules.get("output_port", "output")
                    in_port = rules.get("input_port", "control_image")
                    if target_node and target_node in inner.nodes and target_port:
                        inner.connect(node_name, out_port, target_node, target_port)
                    if graph_input:
                        # Несколько ControlNet: уникальный вход control_image_<node_name>, иначе второй перезапишет первый
                        if type == "adapter/controlnet" and graph_input == "control_image" and graph_input in inner.graph_inputs:
                            graph_input = f"control_image_{node_name}"
                        inner.expose_input(graph_input, node_name, in_port)
                    if role == "adapter" and type == "adapter/controlnet":
                        for inp, port in (
                            ("condition", "encoder_hidden_states"),
                            ("latents", "sample"),
                            ("timestep", "timestep"),
                        ):
                            if inp in inner.graph_inputs:
                                inner.expose_input(inp, node_name, port)
                # Проброс входа цикла на верхний уровень (имя = graph_input, для нескольких ControlNet — control_image_<node_name>)
                if rules and graph_input:
                    self.expose_input(graph_input, target_inner, graph_input)
            self.infer_metadata_if_needed()
            return self

        if node_name in self.nodes:
            node_name = f"{node_name}_{len(self.nodes)}"
        if block is not None:
            self.nodes[node_name] = block

        if auto_connect:
            if ip_adapter_with_encoder:
                self._add_node_ip_adapter_with_encoder(node_name, type, cfg, config)
            elif block is not None:
                role = get_role_for_block_type(type)
                rules = get_connection_rules(role)
                if rules:
                    target_node = rules.get("target_node")
                    target_port = rules.get("target_port")
                    graph_input = rules.get("graph_input")
                    out_port = rules.get("output_port", "output")
                    in_port = rules.get("input_port", "control_image")
                    if target_node and target_node in self.nodes and target_port:
                        self.connect(node_name, out_port, target_node, target_port)
                    if graph_input:
                        self.expose_input(graph_input, node_name, in_port)
            from .pipeline_auto_wire import apply_pipeline_auto_wire
            apply_pipeline_auto_wire(self)

        self.infer_metadata_if_needed()
        return self

    def _add_node_ip_adapter_with_encoder(
        self, adapter_node_name: str, adapter_type: str, adapter_cfg: dict, config: dict
    ) -> None:
        """Add IP-Adapter plus image encoder in one go (BlockBuilder inside)."""
        from yggdrasil.core.block.builder import BlockBuilder
        from .adapters import get_cross_attention_dim_from_graph
        encoder_pretrained = config.get("image_encoder_pretrained") or config.get("pretrained") or "openai/clip-vit-large-patch14"
        encoder = BlockBuilder.build({"type": "conditioner/clip_vision", "pretrained": encoder_pretrained})
        image_embed_dim = getattr(encoder, "embedding_dim", None) or (
            768 if "clip-vit-large" in str(encoder_pretrained) else 1024
        )
        meta = getattr(self, "metadata", None) or {}
        cross_attention_dim = (
            adapter_cfg.get("cross_attention_dim")
            or meta.get("cross_attention_dim")
            or get_cross_attention_dim_from_graph(self)
        )
        if cross_attention_dim is None:
            raise ValueError(
                "IP-Adapter: could not infer cross_attention_dim from graph (backbone not built yet?). "
                "Add the backbone and call graph.to(device) first, or pass cross_attention_dim=... when adding the node."
            )
        ip_cfg = {
            **{k: v for k, v in adapter_cfg.items() if k != "image_encoder_pretrained"},
            "type": adapter_type,
            "image_embed_dim": image_embed_dim,
            "cross_attention_dim": cross_attention_dim,
        }
        ip_adapter = BlockBuilder.build(ip_cfg)
        encoder_name = "ip_image_encoder"
        if encoder_name in self.nodes:
            encoder_name = f"{encoder_name}_{len(self.nodes)}"
        self.nodes[encoder_name] = encoder
        self.nodes[adapter_node_name] = ip_adapter
        self.connect(encoder_name, "embedding", adapter_node_name, "image_features")
        self.expose_input("ip_image", encoder_name, "raw_condition")

    def add_stage(
        self,
        name: str,
        stage: Any = None,
        *,
        template: str | None = None,
        path: str | Path | None = None,
        auto_connect_to_previous: bool = False,
        auto_connect_by_ports: bool = False,
        **template_kwargs,
    ) -> ComputeGraph:
        """Add a pipeline stage (AbstractStage) to the graph.
        
        Use when this ComputeGraph is a pipeline-level graph (nodes = stages).
        Exactly one of: stage, template, or path must be provided.
        
        Args:
            name: Unique stage node name.
            stage: Existing AbstractStage instance (or ComputeGraph to wrap).
            template: Template name → build graph via from_template and wrap in AbstractStage.
            path: Path to YAML → load graph via from_yaml and wrap in AbstractStage.
            auto_connect_to_previous: If True, connect previous node's "output" to this stage's "input".
            auto_connect_by_ports: If True, connect the last stage that has no outgoing "output"
                edge to this stage's "input" (port-based ordering per TZ §4.2). When only one
                such stage exists, equivalent to auto_connect_to_previous.
            **template_kwargs: Passed to from_template when template= is used.
        
        Returns:
            self (for chaining).
        """
        from yggdrasil.core.graph.stage import AbstractStage

        if name in self.nodes:
            raise ValueError(f"Stage '{name}' already exists in graph '{self.name}'")
        inner: ComputeGraph
        if stage is not None:
            if hasattr(stage, "graph"):
                inner = stage.graph
                block = stage
            else:
                inner = stage
                block = AbstractStage(config={"type": "stage/abstract"}, graph=inner)
        elif template is not None:
            inner = ComputeGraph.from_template(template, **template_kwargs)
            block = AbstractStage(config={"type": "stage/abstract"}, graph=inner)
        elif path is not None:
            p = Path(path)
            inner = ComputeGraph.from_yaml(p)
            block = AbstractStage(config={"type": "stage/abstract"}, graph=inner)
        else:
            raise ValueError("add_stage requires one of: stage=, template=, path=")
        self.nodes[name] = block
        if auto_connect_to_previous:
            prev_name = list(self.nodes)[-2] if len(self.nodes) > 1 else None
            if prev_name:
                self.connect(prev_name, "output", name, "input")
        elif auto_connect_by_ports:
            self._connect_stage_by_ports(name)
        return self

    def _connect_stage_by_ports(self, new_name: str) -> None:
        """Connect the new stage to the last stage that has no outgoing 'output' edge (port-based order, TZ §4.2)."""
        nodes_with_output_used = {e.src_node for e in self.edges if e.src_port == "output"}
        candidates = [n for n in self.nodes if n != new_name and n not in nodes_with_output_used]
        if not candidates:
            return
        prev_name = candidates[-1]
        self.connect(prev_name, "output", new_name, "input")
    
    def connect(
        self,
        src: str,
        src_port: str,
        dst: str,
        dst_port: str,
        *,
        validate: bool = True,
    ) -> ComputeGraph:
        """Соединить выходной порт одного узла с входным портом другого.
        
        Args:
            src: Имя узла-источника.
            src_port: Имя выходного порта.
            dst: Имя узла-приёмника.
            dst_port: Имя входного порта.
            validate: Проверять ли совместимость портов (default True).
        
        Returns:
            self (для chaining).
        
        Raises:
            ValueError: Если узлы не найдены или порты несовместимы.
        """
        if src not in self.nodes:
            raise ValueError(f"Source node '{src}' not found in graph '{self.name}'")
        if dst not in self.nodes:
            raise ValueError(f"Destination node '{dst}' not found in graph '{self.name}'")
        
        # Port validation (soft — warns if ports declared but incompatible)
        if validate:
            self._validate_edge(src, src_port, dst, dst_port)
        
        edge = Edge(src_node=src, src_port=src_port, dst_node=dst, dst_port=dst_port)
        self.edges.append(edge)
        return self
    
    def _validate_edge(self, src: str, src_port: str, dst: str, dst_port: str):
        """Validate port compatibility for an edge.
        
        Raises ValueError if validation_mode='strict'.
        Otherwise logs a warning.
        """
        import logging
        _logger = logging.getLogger(__name__)
        
        src_block = self.nodes[src]
        dst_block = self.nodes[dst]
        
        src_ports = getattr(src_block, 'declare_io', lambda: {})()
        dst_ports = getattr(dst_block, 'declare_io', lambda: {})()
        
        if not src_ports or not dst_ports:
            return
        
        sp = src_ports.get(src_port)
        dp = dst_ports.get(dst_port)
        
        if sp is not None and dp is not None:
            from yggdrasil.core.block.port import PortValidator
            valid, msg = PortValidator.validate_connection(
                getattr(src_block, 'block_type', 'unknown'), sp,
                getattr(dst_block, 'block_type', 'unknown'), dp,
            )
            if not valid:
                if self.metadata.get("strict_validation", False):
                    raise ValueError(f"Port incompatible: {msg}")
                _logger.warning(f"Port compatibility warning: {msg}")
        elif sp is None and src_ports:
            msg = f"Output port '{src_port}' not declared on {src} ({getattr(src_block, 'block_type', '?')})"
            if self.metadata.get("strict_validation", False):
                raise ValueError(msg)
            _logger.debug(msg)
        elif dp is None and dst_ports:
            msg = f"Input port '{dst_port}' not declared on {dst} ({getattr(dst_block, 'block_type', '?')})"
            if self.metadata.get("strict_validation", False):
                raise ValueError(msg)
            _logger.debug(msg)
    
    def expose_input(
        self,
        graph_input_name: str,
        node: str,
        port: str,
    ) -> ComputeGraph:
        """Объявить входной порт графа (маппинг на входной порт узла).
        
        При execute(**inputs) значение inputs[graph_input_name]
        будет передано в node.port.
        
        Поддерживает fan-out: один и тот же graph_input_name может быть
        привязан к нескольким (node, port) парам. Повторный вызов с тем же
        именем добавляет новый target, а не перезаписывает старый.
        """
        if node not in self.nodes:
            raise ValueError(f"Node '{node}' not found in graph '{self.name}'")
        
        target = (node, port)
        if graph_input_name in self.graph_inputs:
            # Fan-out: append new target
            targets = self.graph_inputs[graph_input_name]
            if target not in targets:
                targets.append(target)
        else:
            self.graph_inputs[graph_input_name] = [target]
        return self
    
    def expose_output(
        self,
        graph_output_name: str,
        node: str,
        port: str,
    ) -> ComputeGraph:
        """Объявить выходной порт графа (маппинг на выходной порт узла)."""
        if node not in self.nodes:
            raise ValueError(f"Node '{node}' not found in graph '{self.name}'")
        self.graph_outputs[graph_output_name] = (node, port)
        return self
    
    # ==================== LEGO-ОПЕРАЦИИ ====================
    
    def replace_node(self, name: str, new_block: Any, *, validate: bool = True) -> ComputeGraph:
        """Заменить узел, сохраняя все его соединения.

        Поддерживается замена внутреннего узла цикла: name вида "loop_node.inner_name"
        (например "MyAwesomeBackbone.backbone") — тогда заменяется узел inner_name
        во внутреннем графе узла loop_node; цикл денойзинга не пересоздаётся.

        Args:
            name: Имя узла для замены или "outer.inner" для узла внутри loop.
            new_block: Новый блок.
            validate: Проверять ли совместимость портов с существующими соединениями.
        """
        if "." in name:
            outer_name, inner_name = name.split(".", 1)
            if outer_name not in self.nodes:
                raise ValueError(f"Node '{outer_name}' not found in graph '{self.name}'")
            outer_block = self.nodes[outer_name]
            inner = getattr(outer_block, "graph", None) or (
                getattr(outer_block, "_loop", None) and getattr(outer_block._loop, "graph", None)
            )
            if inner is None or not getattr(inner, "nodes", None):
                raise ValueError(f"Node '{outer_name}' has no inner graph (cannot replace '{inner_name}')")
            if inner_name not in inner.nodes:
                raise ValueError(f"Inner node '{inner_name}' not found in '{outer_name}'")
            if validate:
                self._validate_replacement(inner_name, new_block, graph=inner)
            inner.nodes[inner_name] = new_block
            self.infer_metadata_if_needed()
            orch = self._get_orchestrator()
            if orch is not None:
                orch.invalidate()
            return self
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' not found in graph '{self.name}'")
        if validate:
            self._validate_replacement(name, new_block)
        self.nodes[name] = new_block
        self.infer_metadata_if_needed()
        orch = self._get_orchestrator()
        if orch is not None:
            orch.invalidate()
        return self
    
    def _validate_replacement(self, name: str, new_block: Any, graph: Optional["ComputeGraph"] = None):
        """Check that new_block's ports are compatible with existing edges."""
        import logging
        _logger = logging.getLogger(__name__)
        g = graph if graph is not None else self
        edges = getattr(g, "edges", [])
        new_ports = getattr(new_block, "declare_io", lambda: {})()
        if not new_ports:
            return
        for edge in edges:
            if edge.dst_node == name and edge.dst_port not in new_ports:
                _logger.warning(
                    f"Replace warning: new block has no input port '{edge.dst_port}' "
                    f"(required by edge from '{edge.src_node}.{edge.src_port}')"
                )
        for edge in edges:
            if edge.src_node == name and edge.src_port not in new_ports:
                _logger.warning(
                    f"Replace warning: new block has no output port '{edge.src_port}' "
                    f"(required by edge to '{edge.dst_node}.{edge.dst_port}')"
                )
    
    def remove_node(self, name: str) -> ComputeGraph:
        """Удалить узел и все его соединения."""
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' not found in graph '{self.name}'")
        del self.nodes[name]
        self.edges = [
            e for e in self.edges
            if e.src_node != name and e.dst_node != name
        ]
        # Remove graph_inputs that reference this node (fan-out aware)
        new_inputs: Dict[str, List[Tuple[str, str]]] = {}
        for k, targets in self.graph_inputs.items():
            filtered = [(n, p) for n, p in targets if n != name]
            if filtered:
                new_inputs[k] = filtered
        self.graph_inputs = new_inputs
        self.graph_outputs = {
            k: v for k, v in self.graph_outputs.items()
            if v[0] != name
        }
        return self
    
    def insert_between(
        self,
        src: str,
        src_port: str,
        dst: str,
        dst_port: str,
        name: str,
        block: Any,
        in_port: str = "input",
        out_port: str = "output",
    ) -> ComputeGraph:
        """Вставить блок между двумя существующими узлами.
        
        Разрывает ребро src.src_port -> dst.dst_port и вставляет:
        src.src_port -> name.in_port, name.out_port -> dst.dst_port
        """
        # Remove the direct edge
        self.edges = [
            e for e in self.edges
            if not (e.src_node == src and e.src_port == src_port
                    and e.dst_node == dst and e.dst_port == dst_port)
        ]
        self.add_node(name, block)
        self.connect(src, src_port, name, in_port)
        self.connect(name, out_port, dst, dst_port)
        return self
    
    def merge_graph(
        self,
        other: ComputeGraph,
        connections: List[Tuple[str, str, str, str]] | None = None,
        prefix: str = "",
    ) -> ComputeGraph:
        """Объединить с другим графом.
        
        Args:
            other: Граф для объединения.
            connections: Список (src, src_port, dst, dst_port) для соединения графов.
            prefix: Префикс для имён узлов из other (для избежания коллизий).
        """
        for node_name, block in other.nodes.items():
            full_name = f"{prefix}{node_name}" if prefix else node_name
            self.add_node(full_name, block)
        
        for edge in other.edges:
            src = f"{prefix}{edge.src_node}" if prefix else edge.src_node
            dst = f"{prefix}{edge.dst_node}" if prefix else edge.dst_node
            self.connect(src, edge.src_port, dst, edge.dst_port)
        
        if connections:
            for src, sp, dst, dp in connections:
                self.connect(src, sp, dst, dp)
        
        return self
    
    def clone(self) -> ComputeGraph:
        """Глубокая копия графа (блоки не копируются — только ссылки)."""
        new_graph = ComputeGraph(self.name)
        new_graph.nodes = OrderedDict(self.nodes)
        new_graph.edges = list(self.edges)
        # Deep-copy fan-out lists
        new_graph.graph_inputs = {k: list(v) for k, v in self.graph_inputs.items()}
        new_graph.graph_outputs = dict(self.graph_outputs)
        new_graph.metadata = dict(self.metadata)
        new_graph._device = self._device
        new_graph._dtype = self._dtype
        new_graph._deferred = list(self._deferred)
        return new_graph
    
    # ==================== ВАЛИДАЦИЯ ====================
    
    def validate(self, strict: bool = False) -> List[str]:
        """Validate graph correctness.
        
        Checks:
        1. All edges reference existing nodes
        2. All graph_inputs/graph_outputs reference existing nodes
        3. Graph is acyclic (topological sort succeeds)
        4. Port compatibility on all edges
        5. Undeclared ports on edges (if blocks declare I/O)
        
        Args:
            strict: If True, also raise ValueError on first error.
            
        Returns:
            List of error strings (empty = all good).
        """
        errors = []
        warnings = []
        
        # 1. All edges reference existing nodes
        for edge in self.edges:
            if edge.src_node not in self.nodes:
                errors.append(f"Edge references non-existent source node '{edge.src_node}'")
            if edge.dst_node not in self.nodes:
                errors.append(f"Edge references non-existent destination node '{edge.dst_node}'")
        
        # 2. graph_inputs / graph_outputs reference existing nodes
        for input_name, targets in self.graph_inputs.items():
            for node, port in targets:
                if node not in self.nodes:
                    errors.append(f"Graph input '{input_name}' references non-existent node '{node}'")
        
        for output_name, (node, port) in self.graph_outputs.items():
            if node not in self.nodes:
                errors.append(f"Graph output '{output_name}' references non-existent node '{node}'")
        
        # 3. Acyclicity
        try:
            self.topological_sort()
        except ValueError as e:
            errors.append(str(e))
        
        # 4+5. Port validation
        for edge in self.edges:
            src_block = self.nodes.get(edge.src_node)
            dst_block = self.nodes.get(edge.dst_node)
            if src_block is None or dst_block is None:
                continue
            
            src_ports = getattr(src_block, 'declare_io', lambda: {})()
            dst_ports = getattr(dst_block, 'declare_io', lambda: {})()
            
            # Check undeclared ports
            if src_ports and edge.src_port not in src_ports and edge.src_port != "output":
                warnings.append(
                    f"Edge {edge}: source port '{edge.src_port}' not declared by "
                    f"{getattr(src_block, 'block_type', type(src_block).__name__)}"
                )
            if dst_ports and edge.dst_port not in dst_ports:
                warnings.append(
                    f"Edge {edge}: dest port '{edge.dst_port}' not declared by "
                    f"{getattr(dst_block, 'block_type', type(dst_block).__name__)}"
                )
            
            # Check port compatibility
            src_port = src_ports.get(edge.src_port)
            dst_port = dst_ports.get(edge.dst_port)
            
            if src_port and dst_port:
                try:
                    from yggdrasil.core.block.port import PortValidator
                    valid, msg = PortValidator.validate_connection(
                        getattr(src_block, 'block_type', 'unknown'),
                        src_port,
                        getattr(dst_block, 'block_type', 'unknown'),
                        dst_port,
                    )
                    if not valid:
                        warnings.append(msg)
                except Exception:
                    pass  # PortValidator may not be available
        
        all_issues = errors + warnings
        
        if strict and all_issues:
            raise ValueError(
                f"Graph '{self.name}' validation failed:\n" + 
                "\n".join(f"  - {e}" for e in all_issues)
            )
        
        return all_issues
    
    def topological_sort(self) -> List[str]:
        """Топологическая сортировка узлов (порядок выполнения).
        
        Returns:
            Список имён узлов в порядке выполнения.
        
        Raises:
            ValueError: Если граф содержит цикл.
        """
        # Build adjacency list
        in_degree: Dict[str, int] = {name: 0 for name in self.nodes}
        adj: Dict[str, List[str]] = {name: [] for name in self.nodes}
        
        for edge in self.edges:
            if edge.src_node in adj and edge.dst_node in in_degree:
                adj[edge.src_node].append(edge.dst_node)
                in_degree[edge.dst_node] += 1
        
        # Kahn's algorithm
        queue = deque([n for n, d in in_degree.items() if d == 0])
        result: List[str] = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(self.nodes):
            remaining = set(self.nodes.keys()) - set(result)
            raise ValueError(
                f"Graph '{self.name}' contains a cycle involving nodes: {remaining}"
            )
        
        return result
    
    def get_node_dependencies(self, node_name: str) -> Set[str]:
        """Получить все узлы, от которых зависит данный узел (рекурсивно)."""
        deps: Set[str] = set()
        queue = deque([node_name])
        
        while queue:
            current = queue.popleft()
            for edge in self.edges:
                if edge.dst_node == current and edge.src_node not in deps:
                    deps.add(edge.src_node)
                    queue.append(edge.src_node)
        
        return deps
    
    # ==================== СЕРИАЛИЗАЦИЯ ====================
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> ComputeGraph:
        """Загрузить граф из YAML-файла.
        
        Поддерживаются два формата:
        
        1) Один граф блоков (nodes/edges/inputs/outputs).
        2) Combined pipeline: stages + links (каждый узел — AbstractStage).
        
        Формат 1::
        
            name: my_pipeline
            nodes:
              clip:
                block: conditioner/clip_text
                config: {pretrained: openai/clip-vit-large-patch14}
              unet:
                block: backbone/unet2d_condition
                config: {pretrained: stable-diffusion-v1-5}
            edges:
              - [clip.embedding, unet.condition]
            inputs:
              prompt: [clip, text]
            outputs:
              result: [unet, output]
        
        Формат 2 (combined_pipeline)::
        
            kind: combined_pipeline
            name: my_combined
            stages:
              - name: stage0
                template: sd15_txt2img
              - name: stage1
                path: ./stage_upscale.yaml
            links:
              - [stage0, output, stage1, input]
            inputs:
              prompt: [stage0, prompt]
            outputs:
              images: [stage1, output]
        """
        conf = OmegaConf.load(path)
        path_obj = Path(path)
        if conf.get("kind") == "combined_pipeline" or (
            "stages" in conf and isinstance(conf.get("stages"), (list, tuple)) and "nodes" not in conf
        ):
            return cls._from_combined_yaml(OmegaConf.to_container(conf, resolve=True), path_obj)
        return cls._from_single_yaml(conf, path_obj)

    @classmethod
    def _from_combined_yaml(cls, conf: dict, config_path: Path) -> ComputeGraph:
        """Build pipeline graph from combined YAML (stages + links)."""
        from yggdrasil.core.graph.stage import AbstractStage

        name = conf.get("name", "combined_pipeline")
        graph = cls(name)
        graph.metadata = dict(conf.get("metadata", {}))
        stages_conf = conf.get("stages", [])
        for s in stages_conf:
            s = dict(s) if hasattr(s, "items") else s
            stage_name = s.get("name") or s.get("id") or str(len(graph.nodes))
            inner = None
            if "template" in s:
                inner = cls.from_template(str(s["template"]), **dict(s.get("config", {})))
            elif "path" in s:
                p = Path(s["path"])
                if not p.is_absolute():
                    p = config_path.parent / p
                inner = cls.from_yaml(p)
            elif "graph" in s:
                inner = cls._build_from_dict(s["graph"])
            else:
                raise ValueError(f"Stage '{stage_name}' must have 'template', 'path', or 'graph'.")
            stage_block = AbstractStage(config={"type": "stage/abstract"}, graph=inner)
            graph.add_node(stage_name, stage_block)
        for edge_def in conf.get("links", []):
            if hasattr(edge_def, "__getitem__") and len(edge_def) >= 4:
                src_node, src_port, dst_node, dst_port = (
                    str(edge_def[0]), str(edge_def[1]), str(edge_def[2]), str(edge_def[3])
                )
            elif hasattr(edge_def, "__getitem__") and len(edge_def) >= 2:
                src_spec, dst_spec = str(edge_def[0]), str(edge_def[1])
                src_node, src_port = src_spec.split(".", 1) if "." in src_spec else (src_spec, "output")
                dst_node, dst_port = dst_spec.split(".", 1) if "." in dst_spec else (dst_spec, "input")
            else:
                continue
            graph.connect(src_node.strip(), src_port.strip(), dst_node.strip(), dst_port.strip())
        for input_name, mapping in conf.get("inputs", {}).items():
            if isinstance(mapping, (list, tuple)) and len(mapping) >= 2:
                graph.expose_input(input_name, str(mapping[0]), str(mapping[1]))
            elif isinstance(mapping, str) and "." in mapping:
                node, port = mapping.split(".", 1)
                graph.expose_input(input_name, node.strip(), port.strip())
        for output_name, mapping in conf.get("outputs", {}).items():
            if isinstance(mapping, (list, tuple)) and len(mapping) >= 2:
                graph.expose_output(output_name, str(mapping[0]), str(mapping[1]))
            elif isinstance(mapping, str) and "." in mapping:
                node, port = mapping.split(".", 1)
                graph.expose_output(output_name, node.strip(), port.strip())
        return graph

    @classmethod
    def _from_single_yaml(cls, conf: Any, path: Path) -> ComputeGraph:
        """Load single block-graph from YAML (nodes/edges/inputs/outputs)."""
        from yggdrasil.core.block.builder import BlockBuilder

        graph = cls(conf.get("name", "unnamed"))
        graph.metadata = dict(conf.get("metadata", {}))
        
        # Build nodes
        for node_name, node_conf in conf.get("nodes", {}).items():
            block_type = node_conf.get("block") or node_conf.get("type")
            config = dict(node_conf.get("config", {}))
            config["type"] = block_type
            block = BlockBuilder.build(config)
            graph.add_node(node_name, block)
        
        # Build edges: "src.port -> dst.port" or [src.port, dst.port]
        for edge_def in conf.get("edges", []):
            if isinstance(edge_def, str):
                pass  # fall through to string parsing below
            elif hasattr(edge_def, '__getitem__') and len(edge_def) >= 2:
                src_spec, dst_spec = str(edge_def[0]), str(edge_def[1])
                src_node, src_port = src_spec.split(".", 1)
                dst_node, dst_port = dst_spec.split(".", 1)
                graph.connect(src_node.strip(), src_port.strip(), dst_node.strip(), dst_port.strip())
                continue
            else:
                # String format: "src.port -> dst.port"
                parts = str(edge_def).split("->")
                src_spec = parts[0].strip()
                dst_spec = parts[1].strip()
            
            src_node, src_port = str(src_spec).split(".", 1)
            dst_node, dst_port = str(dst_spec).split(".", 1)
            graph.connect(src_node.strip(), src_port.strip(), dst_node.strip(), dst_port.strip())
        
        # Graph inputs (supports fan-out: list of [node, port] pairs)
        for input_name, mapping in conf.get("inputs", {}).items():
            if isinstance(mapping, str):
                node, port = mapping.split(".", 1)
                graph.expose_input(input_name, node.strip(), port.strip())
            elif hasattr(mapping, '__getitem__') and len(mapping) >= 2:
                # Check if it's a list of pairs (fan-out) or a single pair
                first = mapping[0]
                if hasattr(first, '__getitem__') and not isinstance(first, str) and len(first) >= 2:
                    # Fan-out: [[node, port], [node, port], ...]
                    for pair in mapping:
                        graph.expose_input(input_name, str(pair[0]), str(pair[1]))
                else:
                    # Single pair: [node, port]
                    graph.expose_input(input_name, str(mapping[0]), str(mapping[1]))
            else:
                node, port = str(mapping).split(".", 1)
                graph.expose_input(input_name, node.strip(), port.strip())
        
        # Graph outputs
        for output_name, mapping in conf.get("outputs", {}).items():
            if isinstance(mapping, str):
                node, port = mapping.split(".", 1)
                graph.expose_output(output_name, node.strip(), port.strip())
            elif hasattr(mapping, '__getitem__') and len(mapping) >= 2:
                graph.expose_output(output_name, str(mapping[0]), str(mapping[1]))
            else:
                node, port = str(mapping).split(".", 1)
                graph.expose_output(output_name, node.strip(), port.strip())
        
        return graph
    
    def _to_dict(self) -> dict:
        """Serialize graph to dict (for embedding in combined pipeline stages)."""
        data = {"name": self.name, "metadata": dict(self.metadata), "nodes": {}, "edges": [], "inputs": {}, "outputs": {}}
        for node_name, block in self.nodes.items():
            block_type = getattr(block, "block_type", "unknown")
            config = {}
            if hasattr(block, "config"):
                try:
                    config = OmegaConf.to_container(block.config, resolve=True)
                except Exception:
                    config = dict(block.config) if block.config else {}
            data["nodes"][node_name] = {"block": block_type, "config": config}
        for edge in self.edges:
            data["edges"].append([f"{edge.src_node}.{edge.src_port}", f"{edge.dst_node}.{edge.dst_port}"])
        for input_name, targets in self.graph_inputs.items():
            data["inputs"][input_name] = [targets[0][0], targets[0][1]] if len(targets) == 1 else [[n, p] for n, p in targets]
        for output_name, (node, port) in self.graph_outputs.items():
            data["outputs"][output_name] = [node, port]
        return data

    def _is_pipeline_graph(self) -> bool:
        """True if all nodes are AbstractStage (pipeline-level graph)."""
        if not self.nodes:
            return False
        for block in self.nodes.values():
            if getattr(block, "block_type", "") != "stage/abstract" or not hasattr(block, "graph"):
                return False
        return True

    def to_yaml(self, path: str | Path) -> None:
        """Сохранить граф в YAML-файл. Combined pipeline (AbstractStage nodes) → kind: combined_pipeline."""
        if self._is_pipeline_graph():
            data = self._to_combined_dict()
        else:
            data = self._to_dict()
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(data), path)

    def _to_combined_dict(self, parameters: dict | None = None) -> dict:
        """Serialize pipeline graph (AbstractStage nodes) as combined_pipeline format."""
        stages = []
        for node_name, block in self.nodes.items():
            inner = getattr(block, "graph", None)
            if inner is not None:
                stages.append({"name": node_name, "graph": inner._to_dict()})
            else:
                stages.append({"name": node_name})
        links = []
        for edge in self.edges:
            links.append([edge.src_node, edge.src_port, edge.dst_node, edge.dst_port])
        data = {
            "kind": "combined_pipeline",
            "name": self.name,
            "metadata": dict(self.metadata),
            "stages": stages,
            "links": links,
            "inputs": {k: [t[0][0], t[0][1]] if len(t) == 1 else [[n, p] for n, p in t] for k, t in self.graph_inputs.items()},
            "outputs": {k: [v[0], v[1]] for k, v in self.graph_outputs.items()},
        }
        if parameters is not None:
            data["parameters"] = parameters
        return data

    # ==================== WORKFLOW SERIALIZATION ====================
    
    def to_workflow(self, path: str | Path, parameters: dict | None = None) -> None:
        """Save a complete workflow: graph structure + runtime parameters.
        
        This is the ComfyUI-like feature: save everything needed to reproduce
        a generation, then replay it with `from_workflow()`.
        
        Args:
            path: Destination file (.yaml or .json)
            parameters: Runtime parameters to include (prompt, seed, guidance_scale, etc.)
        
        Format::
        
            name: sd15_txt2img
            metadata: {...}
            nodes: {...}
            edges: [...]
            inputs: {...}
            outputs: {...}
            parameters:
              prompt: {text: "a beautiful cat"}
              guidance_scale: 7.5
              seed: 42
              num_steps: 28
        """
        import json
        
        # Pipeline graph (AbstractStage nodes) → combined_pipeline format
        if self._is_pipeline_graph():
            data = self._to_combined_dict(parameters=parameters or {})
        else:
            data = self._to_dict()
            data["parameters"] = parameters or {}
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        suffix = path.suffix.lower()
        if suffix == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            OmegaConf.save(OmegaConf.create(data), path)
    
    @classmethod
    def from_workflow(cls, path: str | Path) -> tuple[ComputeGraph, dict]:
        """Load a complete workflow: reconstructs graph + returns runtime parameters.
        
        Args:
            path: Workflow file (.yaml or .json)
            
        Returns:
            Tuple of (graph, parameters) where parameters is a dict of
            runtime inputs (prompt, seed, etc.) that were saved with the workflow.
        
        Example::
        
            graph, params = ComputeGraph.from_workflow("workflow.yaml")
            pipe = InferencePipeline.from_graph(graph)
            output = pipe(**params)
        """
        import json
        
        path = Path(path)
        suffix = path.suffix.lower()
        
        if suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        else:
            data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        
        # Build graph: support combined pipeline (stages + links) or single graph (nodes + edges)
        if data.get("kind") == "combined_pipeline" or (
            "stages" in data
            and isinstance(data.get("stages"), (list, tuple))
            and not data.get("nodes")
        ):
            graph = cls._from_combined_yaml(data, path.parent)
        else:
            graph = cls._build_from_dict(data)
        
        # Extract parameters
        parameters = data.get("parameters", {})
        
        return graph, parameters
    
    @classmethod
    def _build_from_dict(cls, data: dict) -> ComputeGraph:
        """Build a ComputeGraph from a dict structure."""
        from yggdrasil.core.block.builder import BlockBuilder
        
        graph = cls(data.get("name", "unnamed"))
        graph.metadata = dict(data.get("metadata", {}))
        
        # Build nodes
        for node_name, node_conf in data.get("nodes", {}).items():
            block_type = node_conf.get("block") or node_conf.get("type")
            config = dict(node_conf.get("config", {}))
            config["type"] = block_type
            block = BlockBuilder.build(config)
            graph.add_node(node_name, block)
        
        # Build edges
        for edge_def in data.get("edges", []):
            if isinstance(edge_def, str):
                parts = edge_def.split("->")
                src_spec, dst_spec = parts[0].strip(), parts[1].strip()
            elif isinstance(edge_def, (list, tuple)) and len(edge_def) >= 2:
                src_spec, dst_spec = str(edge_def[0]), str(edge_def[1])
            else:
                continue
            
            src_node, src_port = src_spec.split(".", 1)
            dst_node, dst_port = dst_spec.split(".", 1)
            graph.connect(src_node.strip(), src_port.strip(), 
                         dst_node.strip(), dst_port.strip())
        
        # Graph inputs
        for input_name, mapping in data.get("inputs", {}).items():
            if isinstance(mapping, str):
                node, port = mapping.split(".", 1)
                graph.expose_input(input_name, node.strip(), port.strip())
            elif isinstance(mapping, (list, tuple)):
                first = mapping[0]
                if isinstance(first, (list, tuple)):
                    for pair in mapping:
                        graph.expose_input(input_name, str(pair[0]), str(pair[1]))
                else:
                    graph.expose_input(input_name, str(mapping[0]), str(mapping[1]))
        
        # Graph outputs
        for output_name, mapping in data.get("outputs", {}).items():
            if isinstance(mapping, str):
                node, port = mapping.split(".", 1)
                graph.expose_output(output_name, node.strip(), port.strip())
            elif isinstance(mapping, (list, tuple)):
                graph.expose_output(output_name, str(mapping[0]), str(mapping[1]))
        
        return graph
    
    # ==================== ВИЗУАЛИЗАЦИЯ ====================
    
    def visualize(self) -> str:
        """Сгенерировать Mermaid-диаграмму графа."""
        lines = ["graph LR"]
        
        # Nodes
        for name, block in self.nodes.items():
            block_type = getattr(block, 'block_type', 'unknown')
            label = f'{name}["{name}\\n{block_type}"]'
            lines.append(f"    {label}")
        
        # Graph inputs (fan-out aware)
        for input_name, targets in self.graph_inputs.items():
            safe_id = f"in_{input_name}"
            lines.append(f"    {safe_id}(({input_name}))")
            for node, port in targets:
                lines.append(f"    {safe_id} -->|{port}| {node}")
        
        # Edges
        for edge in self.edges:
            label = f"{edge.src_port} -> {edge.dst_port}"
            lines.append(f'    {edge.src_node} -->|"{label}"| {edge.dst_node}')
        
        # Graph outputs
        for output_name, (node, port) in self.graph_outputs.items():
            safe_id = f"out_{output_name}"
            lines.append(f"    {safe_id}(({output_name}))")
            lines.append(f"    {node} -->|{port}| {safe_id}")
        
        return "\n".join(lines)
    
    # ==================== QUERY ====================
    
    def get_edges_from(self, node_name: str) -> List[Edge]:
        """Получить все исходящие рёбра от узла."""
        return [e for e in self.edges if e.src_node == node_name]
    
    def get_edges_to(self, node_name: str) -> List[Edge]:
        """Получить все входящие рёбра к узлу."""
        return [e for e in self.edges if e.dst_node == node_name]
    
    def get_connected_inputs(self, node_name: str) -> Set[str]:
        """Получить имена входных портов узла, к которым подключены рёбра."""
        return {e.dst_port for e in self.edges if e.dst_node == node_name}
    
    def list_nodes(self) -> List[str]:
        """Список имён всех узлов."""
        return list(self.nodes.keys())
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def __contains__(self, node_name: str) -> bool:
        return node_name in self.nodes
    
    def __getitem__(self, node_name: str) -> Any:
        return self.nodes[node_name]
    
    def __repr__(self) -> str:
        total_targets = sum(len(t) for t in self.graph_inputs.values())
        return (
            f"<ComputeGraph '{self.name}' "
            f"nodes={len(self.nodes)} edges={len(self.edges)} "
            f"inputs={list(self.graph_inputs.keys())}({total_targets} targets) "
            f"outputs={list(self.graph_outputs.keys())}>"
        )
    
    # ==================== CLASS METHODS ====================
    
    @classmethod
    def from_template(
        cls,
        template_name: str,
        *,
        device: Any = None,
        dtype: Any = None,
        **kwargs,
    ) -> ComputeGraph:
        """Создать граф из именованного шаблона.
        
        Args:
            template_name: Имя шаблона ("sd15_txt2img", "flux_txt2img", ...).
            device: Устройство ("cuda", "mps", "cpu"). Если указано,
                    граф сразу переносится на устройство.
            dtype: Тип данных. Если None — выбирается автоматически.
            **kwargs: Доп. параметры шаблона (pretrained, ...).
        
        Returns:
            Готовый ComputeGraph (уже на устройстве, если указан device).
        
        Пример::
        
            graph = ComputeGraph.from_template("sd15_txt2img", device="cuda")
            outputs = graph.execute(prompt="a cat", num_steps=28)
        """
        from yggdrasil.core.graph.templates import get_template
        builder_fn = get_template(template_name)
        graph = builder_fn(**kwargs)
        if device is not None:
            graph.to(device, dtype)
        return graph
    
    def execute(self, *, prompt=None, **kwargs: Any) -> Dict[str, Any]:
        """Execute the graph.
        
        Two modes of operation:
        
        1. **High-level** (convenience — delegates to InferencePipeline):
           Accepts prompt, guidance_scale, num_steps, seed, width, height, etc.
           Auto-prepares noise latents and applies overrides.
        
        2. **Low-level** (raw graph execution):
           Pass ready-made graph inputs directly.
        
        Example::
        
            # High-level (auto noise, auto guidance)
            outputs = graph.execute(prompt="a cat", guidance_scale=7.5, num_steps=28, seed=42)
            
            # Low-level (manual)
            outputs = graph.execute(latents=my_noise, prompt={"text": "a cat"})
        
        Returns:
            Dict with graph outputs (keys = expose_output names).
        """
        from yggdrasil.pipeline import InferencePipeline
        pipe = InferencePipeline.from_graph(self)
        
        # Extract high-level params if present
        guidance_scale = kwargs.pop("guidance_scale", None)
        num_steps = kwargs.pop("num_steps", None)
        width = kwargs.pop("width", None)
        height = kwargs.pop("height", None)
        seed = kwargs.pop("seed", None)
        batch_size = kwargs.pop("batch_size", 1)
        negative_prompt = kwargs.pop("negative_prompt", None)
        
        result = pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            width=width,
            height=height,
            seed=seed,
            batch_size=batch_size,
            negative_prompt=negative_prompt,
            **kwargs,
        )
        return result.raw
    
    def execute_raw(self, **inputs: Any) -> Dict[str, Any]:
        """Execute graph with raw inputs (no InferencePipeline convenience).
        
        Use this for non-diffusion graphs or when you want full control.
        """
        from .executor import GraphExecutor
        return GraphExecutor().execute(self, **inputs)

    def with_controlnet(
        self,
        controlnet_pretrained: str = "lllyasviel/control_v11p_sd15_canny",
        conditioning_scale: float = 1.0,
        control_image_source: Optional[Tuple[str, str]] = None,
        **kwargs: Any,
    ) -> "ComputeGraph":
        """Добавить ControlNet к графу (in-place). Граф должен иметь denoise_loop с backbone, поддерживающим adapter_features.
        control_image_source: (node_name, port_name) — брать control_image с выхода узла по ссылке."""
        from .adapters import add_controlnet_to_graph
        add_controlnet_to_graph(
            self,
            controlnet_pretrained=controlnet_pretrained,
            conditioning_scale=conditioning_scale,
            control_image_source=control_image_source,
            **kwargs,
        )
        return self

    def with_adapter(
        self,
        adapter_type: str,
        input_source: Optional[Tuple[str, str]] = None,
        **kwargs: Any,
    ) -> "ComputeGraph":
        """Добавить любой адаптер к графу (in-place): controlnet, ip_adapter и т.д.

        input_source: (node_name, port_name) — передать вход адаптеру по ссылке на выход узла.
        Иначе добавляется новый вход графа (control_image для controlnet, ip_image для ip_adapter).

        Example::
            graph.with_adapter("controlnet", controlnet_pretrained="lllyasviel/control_v11p_sd15_canny")
            graph.with_adapter("controlnet", input_source=("canny_preprocessor", "output"))
        """
        from .adapters import add_adapter_to_graph
        add_adapter_to_graph(self, adapter_type, input_source=input_source, **kwargs)
        return self

    def with_adapters(
        self,
        *,
        controlnet: bool = True,
        ip_adapter: bool = True,
        t2i_adapter: bool = False,
        controlnet_pretrained: Optional[str] = None,
        conditioning_scale: float = 1.0,
        ip_adapter_scale: float = 0.6,
        fp16: bool = True,
        **kwargs: Any,
    ) -> "ComputeGraph":
        """Add ControlNet and/or IP-Adapter (and optionally T2I-Adapter) to the graph in one call.

        Use this when you want a single \"add node\" step that adds both ControlNet and IP-Adapter.

        Example::
            graph.with_adapters(
                controlnet=True,
                ip_adapter=True,
                t2i_adapter=False,
                controlnet_pretrained="diffusers/controlnet-canny-sdxl-1.0",
            )
        """
        from .adapters import add_optional_adapters_to_graph
        add_optional_adapters_to_graph(
            self,
            controlnet=controlnet,
            t2i_adapter=t2i_adapter,
            ip_adapter=ip_adapter,
            controlnet_pretrained=controlnet_pretrained or (
                "diffusers/controlnet-canny-sdxl-1.0"
                if (self.metadata or {}).get("base_model") == "sdxl"
                else "lllyasviel/control_v11p_sd15_canny"
            ),
            conditioning_scale=conditioning_scale,
            fp16=fp16,
            ip_adapter_scale=ip_adapter_scale,
            **kwargs,
        )
        return self

    def _ensure_ip_adapter_processors_on_device(self) -> None:
        """После переноса графа на устройство — синхронизировать процессоры IP-Adapter с UNet (device/dtype)."""
        try:
            from .adapters import ensure_ip_adapter_processors_on_device
        except ImportError:
            return
        for _name, block in self._iter_all_blocks():
            unet = getattr(block, "unet", None)
            if unet is not None and getattr(unet, "attn_processors", None):
                ensure_ip_adapter_processors_on_device(unet)

    def _iter_all_blocks(self):
        """Итерация по ВСЕМ блокам, включая вложенные SubGraph и блоки с _loop (e.g. DenoiseLoopSDXLBlock)."""
        for name, block in self.nodes.items():
            yield name, block
            if hasattr(block, "graph") and getattr(block, "graph", None) is not None:
                for inner_name, inner_block in block.graph.nodes.items():
                    yield f"{name}.{inner_name}", inner_block
            if hasattr(block, "_loop") and getattr(block, "_loop", None) is not None:
                inner = block._loop
                yield f"{name}._loop", inner
                if hasattr(inner, "graph") and getattr(inner, "graph", None) is not None:
                    for inner_name, inner_block in inner.graph.nodes.items():
                        yield f"{name}.{inner_name}", inner_block

    def list_all_nodes_and_edges(self) -> Tuple[List[str], List[Tuple[str, str, str, str]]]:
        """Собрать все узлы и все соединения (включая вложенные графы).

        Returns:
            (nodes, edges): nodes — список имён узлов (вложенные с префиксом, напр. denoise_loop.controlnet);
            edges — список (src_node, src_port, dst_node, dst_port) по порядку.
        """
        nodes: List[str] = []
        edges: List[Tuple[str, str, str, str]] = []

        def collect(g: "ComputeGraph", prefix: str) -> None:
            for name in g.nodes:
                full = f"{prefix}{name}"
                nodes.append(full)
            for e in g.edges:
                edges.append((f"{prefix}{e.src_node}", e.src_port, f"{prefix}{e.dst_node}", e.dst_port))
            for name, block in g.nodes.items():
                if hasattr(block, "graph") and block.graph is not None:
                    collect(block.graph, f"{prefix}{name}.")

        for name in self.nodes:
            nodes.append(name)
        for e in self.edges:
            edges.append((e.src_node, e.src_port, e.dst_node, e.dst_port))
        for name, block in self.nodes.items():
            if hasattr(block, "graph") and block.graph is not None:
                collect(block.graph, f"{name}.")

        return nodes, edges
