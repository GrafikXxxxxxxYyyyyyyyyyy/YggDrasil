# yggdrasil/core/model/outer_module.py
"""AbstractOuterModule — обёртка над внешней моделью или сервисом.

Не часть диффузионного цикла. Примеры: face recognition, object detector, classifier.
Detailer/Upscaler — AbstractStage, не OuterModule.
"""
from __future__ import annotations

from typing import Any, Dict

from ...core.block.base import AbstractBaseBlock
from ...core.block.registry import register_block
from ...core.block.port import Port, InputPort, OutputPort


@register_block("outer_module/abstract")
class AbstractOuterModule(AbstractBaseBlock):
    """Обёртка над внешней моделью или сервисом.

    Подключается к пайплайну «сбоку», не внутри диффузионного цикла.
    Примеры: face recognition, object detector, classifier, segmenter.
    Detailer- и Upscaler-пайплайны задаются как AbstractStage, не OuterModule.
    """

    block_type = "outer_module/abstract"

    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        return {
            "input": InputPort("input", data_type="any", description="Input (image, video, tensor)"),
            "output": OutputPort("output", data_type="any", description="Output (boxes, masks, embeddings, etc.)"),
        }

    def process(self, **port_inputs) -> Dict[str, Any]:
        """Override: call external model and return output."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement process() for external model call"
        )
