"""DatasetBlock — wraps any dataset as a graph input node.

In a training graph:
    [DatasetBlock] --> [backbone] --> [LossBlock]
                           |               ^
                     [conditioner]    [target from dataset]

The DatasetBlock provides data to the graph at each training step.
"""
from __future__ import annotations

import torch
from typing import Any, Dict, Optional, Iterator
from omegaconf import DictConfig

from ...core.block.base import AbstractBaseBlock
from ...core.block.registry import register_block
from ...core.block.port import InputPort, OutputPort, Port


@register_block("data/dataset")
class DatasetBlock(AbstractBaseBlock):
    """Wraps a dataset as a graph node.
    
    Provides one batch at a time via process().
    The trainer calls set_batch() before each graph execution.
    
    Ports:
        OUT: data, condition, target, latent (all optional, depends on dataset)
    """
    
    block_type = "data/dataset"
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "data/dataset"}
        super().__init__(config)
        self._current_batch: Dict[str, Any] = {}
        self._dataset = None
        self._iterator: Optional[Iterator] = None
    
    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        return {
            "data": OutputPort("data", description="Raw data tensor (pixel/audio/etc)"),
            "condition": OutputPort("condition", data_type="dict", description="Condition dict"),
            "target": OutputPort("target", description="Target tensor for loss"),
            "latent": OutputPort("latent", description="Pre-cached latent"),
        }
    
    def set_dataset(self, dataset):
        """Attach a dataset (AbstractDataSource or torch Dataset)."""
        self._dataset = dataset
        return self
    
    def set_batch(self, batch: Dict[str, Any]):
        """Set the current batch (called by trainer before graph execution)."""
        self._current_batch = batch
    
    def process(self, **port_inputs) -> dict:
        """Output the current batch as port values."""
        out = {}
        for key, value in self._current_batch.items():
            out[key] = value
        # Ensure 'output' exists
        out["output"] = out.get("data", out.get("latent"))
        return out
    
    def _forward_impl(self, **kwargs):
        return self.process(**kwargs)


@register_block("data/image_dataset")
class ImageDatasetBlock(DatasetBlock):
    """Image dataset block — wraps ImageFolderSource.
    
    Config:
        root: str — path to image folder
        resolution: int (default 512)
        center_crop: bool (default True)
    """
    
    block_type = "data/image_dataset"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config)
        root = self.config.get("root")
        if root:
            from ...training.data import ImageFolderSource
            self._dataset = ImageFolderSource(
                root=root,
                resolution=int(self.config.get("resolution", 512)),
                center_crop=bool(self.config.get("center_crop", True)),
            )


@register_block("data/video_dataset")
class VideoDatasetBlock(DatasetBlock):
    """Video dataset block placeholder.
    
    Config:
        root: str — path to video folder
        num_frames: int (default 16)
        resolution: int (default 256)
    """
    
    block_type = "data/video_dataset"


@register_block("data/audio_dataset")
class AudioDatasetBlock(DatasetBlock):
    """Audio dataset block placeholder.
    
    Config:
        root: str — path to audio folder  
        sample_rate: int (default 16000)
        duration: float (default 5.0)
    """
    
    block_type = "data/audio_dataset"
