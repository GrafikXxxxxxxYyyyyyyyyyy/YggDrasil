"""
Abstract task nodes: Backbone, Solver, Codec, Conditioner, Tokenizer, Adapter, Guidance.

Canon: WorldGenerator_2.0/Abstract_Task_Nodes.md §3–9, TODO_02 Part A.
Each inherits AbstractBaseBlock and declares role + canonical ports.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List

from yggdrasill.foundation.block import AbstractBaseBlock
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.roles import (
    ROLE_BACKBONE,
    ROLE_SOLVER,
    ROLE_CODEC,
    ROLE_CONDITIONER,
    ROLE_TOKENIZER,
    ROLE_ADAPTER,
    ROLE_GUIDANCE,
)


class AbstractBackbone(AbstractBaseBlock):
    """
    One step of model prediction: (latent, timestep, condition) -> pred.
    Canon: Abstract_Task_Nodes.md §3.
    """

    def declare_ports(self) -> List[Port]:
        return [
            Port("latent", PortDirection.IN, dtype=PortType.TENSOR),
            Port("timestep", PortDirection.IN, dtype=PortType.TENSOR),
            Port("condition", PortDirection.IN, dtype=PortType.ANY, optional=True),
            Port("pred", PortDirection.OUT, dtype=PortType.TENSOR),
        ]

    @property
    def block_type(self) -> str:
        return ROLE_BACKBONE

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ...


class AbstractSolver(AbstractBaseBlock):
    """
    Один шаг дискретизации по времени: (latent, timestep, pred) -> (next_latent, next_timestep).

    Расписание (noise schedule: sigma, alpha от timestep и т.п.) целиком входит в роль Solver:
    реализуется внутри блока (DDIM, Euler, …) или задаётся конфигом. Отдельного узла/роли
    «NoiseSchedule» нет — Solver выполняет и шаг схемы, и при необходимости расписание.
    Canon: Abstract_Task_Nodes.md §4.
    """

    def declare_ports(self) -> List[Port]:
        return [
            Port("latent", PortDirection.IN, dtype=PortType.TENSOR),
            Port("timestep", PortDirection.IN, dtype=PortType.TENSOR),
            Port("pred", PortDirection.IN, dtype=PortType.TENSOR),
            Port("next_latent", PortDirection.OUT, dtype=PortType.TENSOR),
            Port("next_timestep", PortDirection.OUT, dtype=PortType.TENSOR),
        ]

    @property
    def block_type(self) -> str:
        return ROLE_SOLVER

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ...


class AbstractCodec(AbstractBaseBlock):
    """
    Encode and/or decode (e.g. image <-> latent). Ports for both directions.
    Canon: Abstract_Task_Nodes.md §5.
    """

    def declare_ports(self) -> List[Port]:
        return [
            Port("encode_image", PortDirection.IN, dtype=PortType.IMAGE, optional=True),
            Port("encode_latent", PortDirection.OUT, dtype=PortType.TENSOR),
            Port("decode_latent", PortDirection.IN, dtype=PortType.TENSOR, optional=True),
            Port("decode_image", PortDirection.OUT, dtype=PortType.IMAGE),
        ]

    @property
    def block_type(self) -> str:
        return ROLE_CODEC

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ...


class AbstractConditioner(AbstractBaseBlock):
    """Encode condition (text, image, etc.) into embedding. Canon: §7."""

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, dtype=PortType.ANY),
            Port("embedding", PortDirection.OUT, dtype=PortType.TENSOR),
        ]

    @property
    def block_type(self) -> str:
        return ROLE_CONDITIONER

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ...


class AbstractTokenizer(AbstractBaseBlock):
    """Text -> token_ids. Canon: §8."""

    def declare_ports(self) -> List[Port]:
        return [
            Port("text", PortDirection.IN, dtype=PortType.TEXT),
            Port("token_ids", PortDirection.OUT, dtype=PortType.TENSOR),
            Port("attention_mask", PortDirection.OUT, dtype=PortType.TENSOR, optional=True),
        ]

    @property
    def block_type(self) -> str:
        return ROLE_TOKENIZER

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ...


class AbstractAdapter(AbstractBaseBlock):
    """Adapt signal (LoRA, etc.): (condition, hidden?) -> adapted. Canon: §6."""

    def declare_ports(self) -> List[Port]:
        return [
            Port("condition", PortDirection.IN, dtype=PortType.ANY),
            Port("hidden", PortDirection.IN, dtype=PortType.TENSOR, optional=True),
            Port("adapted", PortDirection.OUT, dtype=PortType.TENSOR),
        ]

    @property
    def block_type(self) -> str:
        return ROLE_ADAPTER

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ...


class AbstractGuidance(AbstractBaseBlock):
    """Correct prediction (CFG, etc.): (pred_cond, pred_uncond, scale?) -> pred_guided. Canon: §9."""

    def declare_ports(self) -> List[Port]:
        return [
            Port("pred_cond", PortDirection.IN, dtype=PortType.TENSOR),
            Port("pred_uncond", PortDirection.IN, dtype=PortType.TENSOR),
            Port("scale", PortDirection.IN, dtype=PortType.TENSOR, optional=True),
            Port("pred_guided", PortDirection.OUT, dtype=PortType.TENSOR),
        ]

    @property
    def block_type(self) -> str:
        return ROLE_GUIDANCE

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ...