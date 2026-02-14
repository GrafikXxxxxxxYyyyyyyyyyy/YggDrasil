#!/usr/bin/env python3
"""YggDrasil — Lego-замена блоков на лету.

Демонстрирует главную фичу фреймворка: можно взять готовый граф
и заменить ЛЮБОЙ блок, сохраняя все соединения.

Все демо работают БЕЗ скачивания весов — используют лёгкие стабы.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from yggdrasil.core.graph.graph import ComputeGraph
from yggdrasil.core.graph.subgraph import LoopSubGraph
from yggdrasil.core.block.builder import BlockBuilder
from yggdrasil.core.block.base import AbstractBlock
from yggdrasil.core.block.port import InputPort, OutputPort, TensorSpec
from yggdrasil.core.block.registry import register_block
from omegaconf import DictConfig


# ==================== Lightweight stubs (no downloads) ====================

@register_block("_stub/backbone")
class StubBackbone(AbstractBlock):
    block_type = "_stub/backbone"
    def __init__(self, config=None):
        super().__init__(config or {"type": "_stub/backbone"})
    @classmethod
    def declare_io(cls):
        return {
            "x": InputPort("x"), "timestep": InputPort("timestep"),
            "condition": InputPort("condition", optional=True),
            "adapter_features": InputPort("adapter_features", optional=True),
            "output": OutputPort("output"),
        }
    def _forward_impl(self, **kw): return kw.get("x", torch.zeros(1))
    def process(self, **kw):
        x = kw.get("x", torch.zeros(1, 4, 8, 8))
        return {"output": x}


@register_block("_stub/conditioner")
class StubConditioner(AbstractBlock):
    block_type = "_stub/conditioner"
    def __init__(self, config=None):
        super().__init__(config or {"type": "_stub/conditioner"})
    @classmethod
    def declare_io(cls):
        return {
            "raw_condition": InputPort("raw_condition"),
            "embedding": OutputPort("embedding"),
        }
    def _forward_impl(self, **kw): return torch.randn(1, 77, 768)
    def process(self, **kw):
        return {"embedding": torch.randn(1, 77, 768), "output": torch.randn(1, 77, 768)}


@register_block("_stub/guidance")
class StubGuidance(AbstractBlock):
    block_type = "_stub/guidance"
    def __init__(self, config=None):
        super().__init__(config or {"type": "_stub/guidance"})
        self.scale = float((config or {}).get("scale", 7.5))
        self._backbone_ref = None
    @classmethod
    def declare_io(cls):
        return {
            "model_output": InputPort("model_output"),
            "x": InputPort("x", optional=True), "t": InputPort("t", optional=True),
            "condition": InputPort("condition", optional=True),
            "guided_output": OutputPort("guided_output"),
        }
    def _forward_impl(self, *a, **kw): return a[0] if a else torch.zeros(1)
    def process(self, **kw):
        out = kw.get("model_output", torch.zeros(1))
        return {"guided_output": out, "output": out}


@register_block("_stub/solver")
class StubSolver(AbstractBlock):
    block_type = "_stub/solver"
    def __init__(self, config=None):
        super().__init__(config or {"type": "_stub/solver"})
    @classmethod
    def declare_io(cls):
        return {
            "model_output": InputPort("model_output"),
            "current_latents": InputPort("current_latents"),
            "timestep": InputPort("timestep"), "next_timestep": InputPort("next_timestep", optional=True),
            "next_latents": OutputPort("next_latents"),
        }
    def _forward_impl(self, **kw): return kw.get("current_latents", torch.zeros(1))
    def process(self, **kw):
        lat = kw.get("current_latents", torch.zeros(1, 4, 8, 8))
        return {"next_latents": lat, "latents": lat, "output": lat}


@register_block("_stub/codec")
class StubCodec(AbstractBlock):
    block_type = "_stub/codec"
    def __init__(self, config=None):
        super().__init__(config or {"type": "_stub/codec"})
    @classmethod
    def declare_io(cls):
        return {
            "latent": InputPort("latent"), "pixel_data": InputPort("pixel_data", optional=True),
            "decoded": OutputPort("decoded"), "encoded": OutputPort("encoded"),
        }
    def _forward_impl(self, **kw): return kw.get("latent", torch.zeros(1))
    def process(self, **kw):
        lat = kw.get("latent", torch.zeros(1, 3, 64, 64))
        return {"decoded": lat, "encoded": lat, "output": lat}


def build_stub_graph(name="sd15_stub", num_steps=10):
    """Собрать полный SD1.5-подобный граф из лёгких стабов."""
    backbone = StubBackbone()
    guidance = StubGuidance({"type": "_stub/guidance", "scale": 7.5})
    solver = StubSolver()
    conditioner = StubConditioner()
    codec = StubCodec()

    guidance._backbone_ref = backbone

    # Inner step graph
    step = ComputeGraph("denoise_step")
    step.add_node("backbone", backbone)
    step.add_node("guidance", guidance)
    step.add_node("solver", solver)

    step.connect("backbone", "output", "guidance", "model_output")
    step.connect("guidance", "guided_output", "solver", "model_output")

    step.expose_input("latents", "backbone", "x")
    step.expose_input("latents", "solver", "current_latents")
    step.expose_input("latents", "guidance", "x")
    step.expose_input("timestep", "backbone", "timestep")
    step.expose_input("timestep", "solver", "timestep")
    step.expose_input("timestep", "guidance", "t")
    step.expose_input("condition", "backbone", "condition")
    step.expose_input("condition", "guidance", "condition")
    step.expose_input("next_timestep", "solver", "next_timestep")
    step.expose_output("next_latents", "solver", "next_latents")
    step.expose_output("latents", "solver", "next_latents")

    loop = LoopSubGraph.create(inner_graph=step, num_iterations=num_steps, show_progress=False)

    graph = ComputeGraph(name)
    graph.metadata = {
        "latent_channels": 4,
        "spatial_scale_factor": 8,
        "default_width": 512,
        "default_height": 512,
        "default_guidance_scale": 7.5,
        "default_num_steps": num_steps,
    }
    graph.add_node("conditioner_0", conditioner)
    graph.add_node("denoise_loop", loop)
    graph.add_node("codec", codec)

    graph.expose_input("prompt", "conditioner_0", "raw_condition")
    graph.connect("conditioner_0", "embedding", "denoise_loop", "condition")
    graph.expose_input("latents", "denoise_loop", "initial_latents")
    graph.connect("denoise_loop", "latents", "codec", "latent")

    graph.expose_output("decoded", "codec", "decoded")
    graph.expose_output("latents", "denoise_loop", "latents")

    return graph


# ==================== 1. Замена solver ====================

def demo_swap_solver():
    """DDIM → Euler: одна строка."""
    print("\n--- 1. Замена solver ---")

    graph = build_stub_graph()
    loop = graph.nodes["denoise_loop"]
    inner = loop.graph

    old = type(inner.nodes["solver"]).__name__
    print(f"  До:    {old}")

    new_solver = BlockBuilder.build({"type": "solver/euler"})
    inner.replace_node("solver", new_solver)

    new = type(inner.nodes["solver"]).__name__
    print(f"  После: {new}")
    print("  ✓ Все соединения сохранены!")


# ==================== 2. Замена guidance ====================

def demo_swap_guidance():
    """Stub guidance → SAG."""
    print("\n--- 2. Замена guidance ---")

    graph = build_stub_graph()
    loop = graph.nodes["denoise_loop"]
    inner = loop.graph

    old = type(inner.nodes["guidance"]).__name__
    print(f"  До:    {old}")

    sag = BlockBuilder.build({"type": "guidance/sag", "scale": 0.75})
    sag._backbone_ref = inner.nodes["backbone"]
    inner.replace_node("guidance", sag)

    new = type(inner.nodes["guidance"]).__name__
    print(f"  После: {new}")
    print("  ✓ Backbone reference установлен!")


# ==================== 3. Изменение параметров ====================

def demo_change_params():
    """Изменить guidance scale на лету."""
    print("\n--- 3. Изменение параметров на лету ---")

    graph = build_stub_graph()
    loop = graph.nodes["denoise_loop"]
    guidance = loop.graph.nodes["guidance"]

    print(f"  До:    scale = {guidance.scale}")
    guidance.scale = 12.0
    print(f"  После: scale = {guidance.scale}")
    print("  ✓ Параметр изменён без пересоздания графа!")


# ==================== 4. Добавление ControlNet ====================

def demo_add_controlnet():
    """Добавить ControlNet в граф."""
    print("\n--- 4. Добавление ControlNet ---")

    graph = build_stub_graph()
    loop = graph.nodes["denoise_loop"]
    inner = loop.graph

    print(f"  Узлы до:    {list(inner.nodes.keys())}")

    # Лёгкий стаб ControlNet
    controlnet = StubBackbone({"type": "_stub/backbone"})
    inner.add_node("controlnet", controlnet)
    inner.expose_input("control_image", "controlnet", "x")
    inner.expose_input("condition", "controlnet", "condition")
    inner.connect("controlnet", "output", "backbone", "adapter_features")

    print(f"  Узлы после: {list(inner.nodes.keys())}")
    print("  ✓ ControlNet добавлен и подключён!")


# ==================== 5. Инспекция графа ====================

def demo_inspect():
    """Показать полную структуру."""
    print("\n--- 5. Инспекция графа ---")

    graph = build_stub_graph()

    print(f"  Граф: {graph.name}")
    print(f"  Узлы ({len(graph.nodes)}):")
    for name, block in graph.nodes.items():
        btype = getattr(block, "block_type", type(block).__name__)
        print(f"    {name}: {btype}")

    print(f"\n  Входы:")
    for name, targets in graph.graph_inputs.items():
        for node, port in targets:
            print(f"    {name} → {node}.{port}")

    print(f"\n  Выходы:")
    for name, (node, port) in graph.graph_outputs.items():
        print(f"    {name} ← {node}.{port}")

    print(f"\n  Рёбра ({len(graph.edges)}):")
    for e in graph.edges:
        print(f"    {e.src_node}.{e.src_port} → {e.dst_node}.{e.dst_port}")

    # Inner graph
    loop = graph.nodes["denoise_loop"]
    inner = loop.graph
    print(f"\n  Inner graph (denoise_step):")
    for name, targets in inner.graph_inputs.items():
        targets_str = ", ".join(f"{n}.{p}" for n, p in targets)
        print(f"    input '{name}' → [{targets_str}]")


# ==================== 6. Выполнение графа ====================

def demo_execute():
    """Полное выполнение графа со стабами."""
    print("\n--- 6. Выполнение графа (стабы) ---")

    graph = build_stub_graph(num_steps=3)
    # Metadata для авто-генерации шума
    graph.metadata = {
        "latent_channels": 4,
        "spatial_scale_factor": 8,
        "default_width": 64,
        "default_height": 64,
    }

    # Новый API: prompt как строка, noise генерируется автоматически
    outputs = graph.execute(
        prompt="a cat",
        num_steps=3,
        seed=42,
        width=64,
        height=64,
    )

    print(f"  Выходные ключи:   {list(outputs.keys())}")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
    print("  ✓ Граф выполнен успешно!")


# ==================== 7. YAML roundtrip ====================

def demo_yaml():
    """Сериализация и десериализация."""
    print("\n--- 7. YAML roundtrip ---")

    graph = build_stub_graph()
    yaml_path = Path("output") / "sd15_stub_graph.yaml"
    yaml_path.parent.mkdir(exist_ok=True)

    graph.to_yaml(yaml_path)
    print(f"  Сохранён:  {yaml_path}")

    loaded = ComputeGraph.from_yaml(yaml_path)
    print(f"  Загружен:  {loaded}")
    print(f"  Узлы:      {list(loaded.nodes.keys())}")
    print(f"  Входы:     {list(loaded.graph_inputs.keys())}")
    print("  ✓ Roundtrip OK!")


# ==================== 8. Все шаблоны ====================

def demo_templates():
    """Список доступных шаблонов."""
    print("\n--- 8. Доступные шаблоны ---")

    from yggdrasil.core.graph.templates import list_templates
    templates = list_templates()
    print(f"  Всего шаблонов: {len(templates)}")
    for t in templates:
        print(f"    - {t}")


# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    demo_templates()
    demo_inspect()
    demo_swap_solver()
    demo_swap_guidance()
    demo_change_params()
    demo_add_controlnet()
    demo_execute()
    demo_yaml()
    print("\n✓ Все демо завершены!")
