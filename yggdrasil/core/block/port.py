# yggdrasil/core/block/port.py
"""Port System — типизированные I/O точки для блоков.

Порт описывает, что блок принимает на вход и что отдаёт на выход.
Это основа dataflow-графа: блоки соединяются через порты.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal, Any, Dict, Tuple, Union

import torch


# ---------------------------------------------------------------------------
# TensorSpec — контракт на тензор
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorSpec:
    """Описывает ожидаемые свойства тензора, проходящего через порт.
    
    Все поля опциональны: ``None`` означает «любое значение допустимо».
    
    Примеры:
        Изображение в пиксельном пространстве::
        
            TensorSpec(ndim=4, channels=3, space="pixel", modality="image")
        
        Латент SD 1.5::
        
            TensorSpec(ndim=4, channels=4, space="latent", modality="image")
        
        Эмбеддинг текста::
        
            TensorSpec(ndim=3, space="embedding", modality="text")
    """
    ndim: Optional[int] = None
    channels: Optional[int] = None
    dtype: Optional[torch.dtype] = None
    space: str = "any"       # "pixel" | "latent" | "embedding" | "noise" | "scalar" | "any"
    modality: str = "any"    # "image" | "audio" | "video" | "3d" | "molecular" | "text" | "timeseries" | "any"

    def is_compatible(self, other: TensorSpec) -> bool:
        """Проверяет совместимость двух спецификаций (output -> input)."""
        if self.ndim is not None and other.ndim is not None and self.ndim != other.ndim:
            return False
        if self.channels is not None and other.channels is not None and self.channels != other.channels:
            return False
        if self.space != "any" and other.space != "any" and self.space != other.space:
            return False
        if self.modality != "any" and other.modality != "any" and self.modality != other.modality:
            return False
        return True


# ---------------------------------------------------------------------------
# Port — типизированная I/O точка
# ---------------------------------------------------------------------------

@dataclass
class Port:
    """Типизированная I/O точка блока.
    
    Каждый блок объявляет свои порты через ``declare_io()``.
    Порт содержит всю информацию для валидации соединений в графе.
    
    Attributes:
        name: Уникальное имя порта внутри блока.
        direction: ``"input"`` или ``"output"``.
        data_type: Тип данных — ``"tensor"``, ``"dict"``, ``"list"``,
                   ``"state"``, ``"scalar"``, ``"any"``.
        spec: Спецификация тензора (только для data_type="tensor").
        optional: Если ``True``, порт может не быть подключён.
        multiple: Если ``True``, порт принимает несколько соединений
                  (результаты собираются в список).
        description: Человекочитаемое описание (для UI и документации).
    """
    name: str
    direction: Literal["input", "output"]
    data_type: str = "tensor"
    spec: Optional[TensorSpec] = None
    optional: bool = False
    multiple: bool = False
    description: str = ""

    def is_compatible_with(self, other: Port) -> bool:
        """Проверяет, можно ли соединить этот output-порт с другим input-портом.
        
        Правила:
        - Направления: self=output, other=input
        - data_type: должны совпадать, или один из них "any"
        - TensorSpec: если оба определены, проверяем совместимость
        """
        if self.direction != "output" or other.direction != "input":
            return False

        # data_type check
        if self.data_type != "any" and other.data_type != "any":
            if self.data_type != other.data_type:
                return False

        # TensorSpec check (only if both have specs)
        if self.spec is not None and other.spec is not None:
            if not self.spec.is_compatible(other.spec):
                return False

        return True


# ---------------------------------------------------------------------------
# Утилиты для быстрого создания портов
# ---------------------------------------------------------------------------

def InputPort(
    name: str,
    data_type: str = "tensor",
    spec: Optional[TensorSpec] = None,
    optional: bool = False,
    multiple: bool = False,
    description: str = "",
) -> Port:
    """Фабрика для создания входного порта."""
    return Port(
        name=name,
        direction="input",
        data_type=data_type,
        spec=spec,
        optional=optional,
        multiple=multiple,
        description=description,
    )


def OutputPort(
    name: str,
    data_type: str = "tensor",
    spec: Optional[TensorSpec] = None,
    description: str = "",
) -> Port:
    """Фабрика для создания выходного порта."""
    return Port(
        name=name,
        direction="output",
        data_type=data_type,
        spec=spec,
        description=description,
    )


# ---------------------------------------------------------------------------
# PortValidator — валидация соединений
# ---------------------------------------------------------------------------

class PortValidator:
    """Валидатор соединений между портами."""

    @staticmethod
    def validate_connection(
        src_block_type: str,
        src_port: Port,
        dst_block_type: str,
        dst_port: Port,
    ) -> Tuple[bool, str]:
        """Проверяет валидность соединения.
        
        Returns:
            (is_valid, error_message)
        """
        if src_port.direction != "output":
            return False, f"Source port '{src_port.name}' on '{src_block_type}' is not an output port"

        if dst_port.direction != "input":
            return False, f"Destination port '{dst_port.name}' on '{dst_block_type}' is not an input port"

        if not src_port.is_compatible_with(dst_port):
            return False, (
                f"Incompatible ports: "
                f"'{src_block_type}.{src_port.name}' (data_type={src_port.data_type}) -> "
                f"'{dst_block_type}.{dst_port.name}' (data_type={dst_port.data_type})"
            )

        return True, ""

    @staticmethod
    def check_required_inputs(
        block_type: str,
        ports: Dict[str, Port],
        connected_inputs: set,
    ) -> list[str]:
        """Проверяет, что все обязательные входные порты подключены.
        
        Returns:
            Список ошибок (пустой = всё ок).
        """
        errors = []
        for port_name, port in ports.items():
            if port.direction == "input" and not port.optional:
                if port_name not in connected_inputs:
                    errors.append(
                        f"Required input port '{port_name}' on '{block_type}' is not connected"
                    )
        return errors
