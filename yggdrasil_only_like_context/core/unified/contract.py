# yggdrasil/core/unified/contract.py
"""Унифицированный контракт пайплайна — канонические имена портов и шаги.

Позволяет:
- Узнать по графу, какие входы/выходы он экспортирует (prompt, control_image, ...).
- Сопоставить узлы графа с абстрактными шагами (condition, denoise_loop, decode, post_process).
- Строить динамический UI (Gradio) и API по любому графу.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .modality import Modality
from .steps import DiffusionStep


# Канонические имена входов графа (унифицированный API)
CANONICAL_INPUTS = {
    "prompt", "negative_prompt", "condition", "uncond",
    "initial_latents", "timesteps", "num_inference_steps", "guidance_scale",
    "control_image", "control_signal", "control_scale",
    "source_image", "strength",  # img2img
    "audio", "video", "mask", "encoder_hidden_states",
}
# Канонические имена выходов
CANONICAL_OUTPUTS = {"latents", "images", "output", "output_signal", "enhanced"}


@dataclass
class StepMapping:
    """Соответствие узла графа абстрактному шагу."""
    step: DiffusionStep
    node_name: str
    block_type: str = ""
    description: str = ""


@dataclass
class PipelineContract:
    """Унифицированное описание пайплайна: входы, выходы, шаги, модальность.

    Строится по ComputeGraph. Используется для:
    - генерации Gradio UI по графу,
    - валидации вызова pipe(**kwargs),
    - документирования и сериализации в API.
    """
    name: str = ""
    modality: Modality = Modality.ANY
    # Графовые входы: имя -> (node, port) или список (node, port) для fan-out
    inputs: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    # Графовые выходы: имя -> (node, port)
    outputs: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    # Узлы, соответствующие каноническим шагам
    step_mappings: List[StepMapping] = field(default_factory=list)
    # Метаданные из graph.metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_input_names(self) -> List[str]:
        return list(self.inputs.keys())

    def get_output_names(self) -> List[str]:
        return list(self.outputs.keys())

    def get_step_node(self, step: DiffusionStep) -> Optional[str]:
        for m in self.step_mappings:
            if m.step == step:
                return m.node_name
        return None

    def has_control(self) -> bool:
        return self.get_step_node(DiffusionStep.CONTROL) is not None or "control_image" in self.inputs

    def has_post_process(self) -> bool:
        return self.get_step_node(DiffusionStep.POST_PROCESS) is not None


def infer_contract(graph: Any) -> PipelineContract:
    """Построить контракт по ComputeGraph.

    Анализирует graph_inputs, graph_outputs, nodes и метаданные,
    пытается сопоставить узлы с DiffusionStep по block_type и именам.
    """
    from ..graph.graph import ComputeGraph

    if not isinstance(graph, ComputeGraph):
        raise TypeError("Expected ComputeGraph")

    contract = PipelineContract(
        name=graph.name,
        metadata=dict(graph.metadata),
    )
    contract.inputs = {k: list(v) if isinstance(v, list) else [v] for k, v in graph.graph_inputs.items()}
    contract.outputs = dict(graph.graph_outputs)

    # Модальность из метаданных
    mod = graph.metadata.get("modality", graph.metadata.get("task"))
    if mod:
        try:
            contract.modality = Modality(str(mod).lower())
        except ValueError:
            pass

    # Эвристика: сопоставить узлы с шагами по block_type и имени
    for node_name, block in graph.nodes.items():
        block_type = getattr(block, "block_type", "") or ""
        step = None
        if "conditioner" in block_type or "conditioner" in node_name or "clip" in node_name:
            step = DiffusionStep.CONDITION
        elif "controlnet" in block_type or "control" in node_name or "t2i_adapter" in block_type:
            step = DiffusionStep.CONTROL
        elif "codec" in block_type or "vae" in block_type or "encodec" in block_type:
            # Может быть encode или decode в зависимости от связей
            if "encode" in node_name:
                step = DiffusionStep.ENCODE
            else:
                step = DiffusionStep.DECODE
        elif "denoise" in node_name or "loop" in node_name:
            step = DiffusionStep.DENOISE_LOOP
        elif "backbone" in block_type or "unet" in block_type or "dit" in block_type:
            step = DiffusionStep.REVERSE_STEP
        elif "solver" in block_type:
            step = DiffusionStep.REVERSE_STEP
        elif "post_process" in block_type or "postprocess" in block_type or "detailer" in node_name or "upscaler" in node_name:
            step = DiffusionStep.POST_PROCESS
        elif "loss" in block_type:
            step = DiffusionStep.LOSS
        if step is not None:
            contract.step_mappings.append(StepMapping(step=step, node_name=node_name, block_type=block_type))

    return contract
