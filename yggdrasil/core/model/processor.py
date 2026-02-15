# yggdrasil/core/model/processor.py
"""AbstractProcessor — пред- и постобработка данных.

Pre/post-processing: resize, normalize, crop, to tensor, etc.
"""
from __future__ import annotations

from typing import Any, Dict

from ...core.block.base import AbstractBaseBlock
from ...core.block.registry import register_block
from ...core.block.port import Port, InputPort, OutputPort


@register_block("processor/abstract")
class AbstractProcessor(AbstractBaseBlock):
    """Пред- и постобработка данных (pre/post-processing).

    Вход: сырые или промежуточные данные (изображение, видео, аудио, тензор).
    Выход: данные в формате для следующего блока или стадии.
    Примеры: resize, normalize, crop, to tensor; финальная нормализация выхода.
    """

    block_type = "processor/abstract"

    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        return {
            "input": InputPort("input", data_type="any", description="Raw or intermediate data"),
            "output": OutputPort("output", data_type="any", description="Processed data"),
        }

    def process(self, **port_inputs) -> Dict[str, Any]:
        """Override: apply preprocessing or postprocessing."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement process() for pre/post-processing"
        )
