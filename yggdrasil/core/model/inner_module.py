# yggdrasil/core/model/inner_module.py
"""AbstractInnerModule — модуль, встраиваемый в процесс деноайзинга.

Контроль внутри цикла обратной диффузии (ControlNet, T2I-Adapter и т.д.).
Выход идёт в backbone.adapter_features.
"""
from __future__ import annotations

from typing import Any, Dict

from ...core.block.base import AbstractBaseBlock
from ...core.block.registry import register_block
from ...core.block.port import Port, InputPort, OutputPort, TensorSpec


@register_block("inner_module/abstract")
class AbstractInnerModule(AbstractBaseBlock):
    """Модуль, встраиваемый в сам процесс деноайзинга.

    Работает внутри цикла обратной диффузии на каждом шаге solver.
    Выход подаётся в backbone.adapter_features.
    Примеры: ControlNet, T2I-Adapter, MotionAdapter (AnimateDiff).
    """

    block_type = "inner_module/abstract"

    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        return {
            "x": InputPort("x", spec=TensorSpec(space="latent"), description="Latent input"),
            "timestep": InputPort("timestep", data_type="tensor", description="Timestep"),
            "condition": InputPort("condition", data_type="dict", optional=True, description="Condition embeddings"),
            "control_image": InputPort("control_image", optional=True, description="Control signal (image, mask, etc.)"),
            "adapter_features": OutputPort(
                "adapter_features",
                data_type="any",
                description="Features to inject into backbone.adapter_features",
            ),
        }

    def process(self, **port_inputs) -> Dict[str, Any]:
        """Override: produce adapter_features from x, timestep, condition, control_image."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement process() to produce adapter_features"
        )
