# yggdrasil/core/model/postprocess.py
"""Абстрактный пост-процессор — улучшение результата после декода (FaceDetailer, upscaler, etc.)."""

from __future__ import annotations

from typing import Any, Dict

from ...core.block.base import AbstractBaseBlock
from ...core.block.registry import register_block
from ...core.block.port import Port, InputPort, OutputPort, TensorSpec


@register_block("postprocess/abstract")
class AbstractPostProcessor(AbstractBaseBlock):
    """Базовый пост-процессор: decoded_signal -> enhanced_signal.

    Подключается после DECODE в унифицированном пайплайне.
    Примеры: FaceDetailer, Real-ESRGAN, кастомный ретушер.
    Модальность-независимый контракт: вход и выход — один и тот же тип сигнала
    (изображение -> изображение, видео -> видео).
    """

    block_type = "postprocess/abstract"

    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        return {
            "input_signal": InputPort(
                "input_signal",
                data_type="tensor",
                spec=TensorSpec(space="pixel", modality="any"),
                description="Decoded output from codec (image, video, audio, ...)",
            ),
            "output_signal": OutputPort(
                "output_signal",
                data_type="tensor",
                spec=TensorSpec(space="pixel", modality="any"),
                description="Enhanced output",
            ),
        }

    def process(self, **port_inputs: Any) -> Dict[str, Any]:
        x = port_inputs.get("input_signal", port_inputs.get("decoded"))
        out = self.apply(x, **{k: v for k, v in port_inputs.items() if k not in ("input_signal", "decoded")})
        return {"output_signal": out, "output": out}

    def apply(self, x, **kwargs) -> Any:
        """Override: input_signal -> enhanced_signal."""
        raise NotImplementedError(f"{type(self).__name__} must implement apply()")
