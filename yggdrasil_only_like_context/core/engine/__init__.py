"""YggDrasil Engine — оркестратор всего процесса (генерация + обучение)."""

from .state import DiffusionState
from .sampler import DiffusionSampler

__all__ = [
    "DiffusionState",
    "DiffusionSampler",
]

# Lazy imports for circular-dependency-prone modules
def __getattr__(name):
    if name == "AbstractPipeline":
        from .pipeline import AbstractPipeline
        return AbstractPipeline
    if name == "SamplingLoop":
        from .loop import SamplingLoop
        return SamplingLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")