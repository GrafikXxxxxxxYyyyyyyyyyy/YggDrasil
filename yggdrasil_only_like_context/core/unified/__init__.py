# yggdrasil/core/unified/__init__.py
"""Унифицированное ядро диффузии — модальность-независимые шаги и контракт пайплайна.

Использование:
    from yggdrasil.core.unified import (
        Modality, DiffusionStep,
        PipelineContract, infer_contract, CANONICAL_INPUTS, CANONICAL_OUTPUTS,
    )
    contract = infer_contract(my_graph)
    if contract.has_control():
        # показывать поле control_image в UI
    ...
"""
from __future__ import annotations

from .modality import (
    Modality,
    SignalSpace,
    MODALITY_DEFAULT_SPACES,
    get_modality_from_spec,
)
from .steps import (
    DiffusionStep,
    CANONICAL_INFERENCE_STEPS,
    CANONICAL_TRAINING_STEPS,
)
from .contract import (
    PipelineContract,
    StepMapping,
    infer_contract,
    CANONICAL_INPUTS,
    CANONICAL_OUTPUTS,
)

__all__ = [
    "Modality",
    "SignalSpace",
    "MODALITY_DEFAULT_SPACES",
    "get_modality_from_spec",
    "DiffusionStep",
    "CANONICAL_INFERENCE_STEPS",
    "CANONICAL_TRAINING_STEPS",
    "PipelineContract",
    "StepMapping",
    "infer_contract",
    "CANONICAL_INPUTS",
    "CANONICAL_OUTPUTS",
]
