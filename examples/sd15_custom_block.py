#!/usr/bin/env python3
"""YggDrasil — Создание кастомного блока и встраивание в граф SD 1.5.

Показывает как написать свой блок с нуля и подключить его
в существующий pipeline одной заменой.

Пример: кастомный Guidance, который комбинирует CFG + edge-aware sharpening.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from typing import Dict, Any, Optional

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.model.guidance import AbstractGuidance
from yggdrasil.core.block.port import InputPort, OutputPort, TensorSpec


# ==================== КАСТОМНЫЙ БЛОК ====================

@register_block("guidance/cfg_sharp")
class CFGWithSharpening(AbstractGuidance):
    """Кастомный Guidance: CFG + laplacian edge sharpening.

    Это пример того, как написать свой блок и подключить его
    в любой pipeline вместо стандартного CFG.

    Формула:
        guided = uncond + scale * (cond - uncond)
        sharpened = guided - sharpen_strength * laplacian(guided)
    """

    block_type = "guidance/cfg_sharp"

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.scale = float(config.get("scale", 7.5))
        self.sharpen_strength = float(config.get("sharpen_strength", 0.3))
        self._backbone_ref = None

    def _forward_impl(self, *args, **kwargs):
        return args[0] if args else torch.zeros(1)

    @classmethod
    def declare_io(cls) -> dict:
        return {
            "model_output": InputPort("model_output", spec=TensorSpec(space="latent"),
                                      description="Conditional model output"),
            "uncond_output": InputPort("uncond_output", optional=True,
                                       description="Unconditional output (explicit)"),
            "x": InputPort("x", optional=True, description="Current latents"),
            "t": InputPort("t", data_type="tensor", optional=True, description="Timestep"),
            "condition": InputPort("condition", data_type="any", optional=True, description="Condition"),
            "guided_output": OutputPort("guided_output", spec=TensorSpec(space="latent"),
                                        description="Guided + sharpened output"),
        }

    def process(self, **port_inputs) -> dict:
        model_output = port_inputs.get("model_output")

        if model_output is None or self.scale <= 1.0:
            return {"guided_output": model_output, "output": model_output}

        # 1. CFG: получаем uncond через backbone_ref или порт
        uncond_output = port_inputs.get("uncond_output")

        if uncond_output is None and self._backbone_ref is not None:
            x = port_inputs.get("x")
            t = port_inputs.get("t")
            condition = port_inputs.get("condition")
            if x is not None and t is not None:
                null_cond = self._make_null_condition(condition)
                with torch.no_grad():
                    uncond_result = self._backbone_ref.process(x=x, timestep=t, condition=null_cond)
                    uncond_output = uncond_result.get("output")

        if uncond_output is not None:
            guided = uncond_output + self.scale * (model_output - uncond_output)
        else:
            guided = model_output

        # 2. Laplacian sharpening
        if self.sharpen_strength > 0 and guided.dim() == 4:
            guided = self._sharpen(guided)

        return {"guided_output": guided, "output": guided}

    def _sharpen(self, x: torch.Tensor) -> torch.Tensor:
        """Laplacian sharpening: x - strength * laplacian(x)."""
        # Laplacian kernel
        kernel = torch.tensor(
            [[0, -1, 0],
             [-1, 4, -1],
             [0, -1, 0]],
            dtype=x.dtype, device=x.device,
        ).unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(x.shape[1], -1, -1, -1)  # per-channel

        laplacian = F.conv2d(x, kernel, padding=1, groups=x.shape[1])
        return x - self.sharpen_strength * laplacian

    def __call__(self, model_output, **kwargs):
        """Legacy mode — просто passthrough."""
        return model_output

    def _make_null_condition(self, condition):
        if condition is None:
            return None
        if isinstance(condition, torch.Tensor):
            return torch.zeros_like(condition)
        if isinstance(condition, dict):
            return {
                k: torch.zeros_like(v) if isinstance(v, torch.Tensor)
                else ("" if isinstance(v, str) else v)
                for k, v in condition.items()
            }
        return None


# ==================== ИСПОЛЬЗОВАНИЕ ====================

def main():
    # Import the stub graph builder from lego_swap example for instant demo
    # (no weight downloads needed)
    sys.path.insert(0, str(Path(__file__).parent))
    from sd15_lego_swap import build_stub_graph

    print("=== Кастомный блок: CFG + Sharpening ===\n")

    # 1. Создаём граф из лёгких стабов (мгновенно, без скачивания весов)
    graph = build_stub_graph(num_steps=10)

    loop = graph.nodes["denoise_loop"]
    inner = loop.graph

    old = type(inner.nodes["guidance"]).__name__
    print(f"Стандартный guidance: {old}")

    # 2. Создаём кастомный блок
    custom = CFGWithSharpening({
        "type": "guidance/cfg_sharp",
        "scale": 7.5,
        "sharpen_strength": 0.3,
    })

    # Устанавливаем backbone ref для внутреннего dual-pass
    custom._backbone_ref = inner.nodes["backbone"]

    # 3. Заменяем guidance одной строкой
    inner.replace_node("guidance", custom)

    new = type(inner.nodes["guidance"]).__name__
    print(f"Кастомный guidance:  {new}")

    # 4. Проверяем что граф валиден
    errors = inner.validate()
    if errors:
        print(f"Ошибки: {errors}")
    else:
        print("Граф валиден! Кастомный блок подключён.")

    # 5. Показываем порты кастомного блока
    ports = custom.declare_io()
    print(f"\nПорты кастомного блока:")
    for name, port in ports.items():
        direction = "IN" if port.direction == "input" else "OUT"
        opt = " (optional)" if getattr(port, "optional", False) else ""
        print(f"  [{direction}] {name}{opt}: {port.description}")

    # 6. Показываем что граф можно выполнить (без реальных весов)
    print(f"\nГраф готов к выполнению:")
    print(f"  Узлы:  {list(graph.nodes.keys())}")
    print(f"  Входы: {list(graph.graph_inputs.keys())}")

    print("\nДля реальной генерации:")
    print('  graph = ComputeGraph.from_template("sd15_txt2img", device="cuda")')
    print("  # ... replace guidance as above ...")
    print('  outputs = graph.execute(prompt="your prompt", num_steps=28, seed=42)')


if __name__ == "__main__":
    main()
