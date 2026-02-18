"""MultiModalAssembler -- chain pipelines across modalities."""
from __future__ import annotations

import torch
from typing import Dict, Any, List, Optional, Callable
from omegaconf import DictConfig, OmegaConf

from ..core.model.modular import ModularDiffusionModel
from ..core.engine.sampler import DiffusionSampler
from ..core.block.builder import BlockBuilder
from ..core.block.registry import auto_discover


class PipelineStage:
    """A single stage in a multi-modal pipeline."""
    
    def __init__(
        self,
        name: str,
        sampler: DiffusionSampler,
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
        condition_keys: Optional[List[str]] = None,
    ):
        self.name = name
        self.sampler = sampler
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.condition_keys = condition_keys or ["prompt"]
    
    def run(self, condition: Dict[str, Any], **kwargs) -> torch.Tensor:
        """Run this pipeline stage."""
        if self.input_transform:
            condition = self.input_transform(condition)
        
        result = self.sampler.sample(condition=condition, **kwargs)
        
        if self.output_transform:
            result = self.output_transform(result)
        
        return result


class MultiModalAssembler:
    """Chains multiple diffusion pipelines for cross-modality generation.
    
    Examples:
    - text -> image -> video (generate image then animate it)
    - text -> image -> upscale -> face-detail (cascaded refinement)
    - text -> audio -> spectrogram visualization
    """
    
    def __init__(self):
        self.stages: List[PipelineStage] = []
    
    def add_stage(
        self,
        name: str,
        sampler: DiffusionSampler,
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
        condition_keys: Optional[List[str]] = None,
    ) -> "MultiModalAssembler":
        """Add a pipeline stage.
        
        Args:
            name: Stage name for logging
            sampler: The DiffusionSampler for this stage
            input_transform: Transform previous output to this stage's condition
            output_transform: Transform this stage's output for the next stage
            condition_keys: Which keys from condition dict this stage uses
            
        Returns:
            Self for chaining
        """
        stage = PipelineStage(
            name=name,
            sampler=sampler,
            input_transform=input_transform,
            output_transform=output_transform,
            condition_keys=condition_keys,
        )
        self.stages.append(stage)
        return self
    
    @torch.no_grad()
    def run(
        self,
        condition: Dict[str, Any],
        return_intermediates: bool = False,
        callback: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run the full multi-modal pipeline.
        
        Args:
            condition: Initial condition dict
            return_intermediates: Whether to return results from all stages
            callback: Called after each stage with (stage_name, result)
            
        Returns:
            Dict with 'result' and optionally 'intermediates'
        """
        intermediates = {}
        current_condition = condition.copy()
        result = None
        
        for i, stage in enumerate(self.stages):
            print(f"[MultiModal] Running stage {i+1}/{len(self.stages)}: {stage.name}")
            
            result = stage.run(current_condition, **kwargs)
            intermediates[stage.name] = result
            
            if callback:
                callback(stage.name, result)
            
            # Feed output into next stage's condition
            current_condition["prev_output"] = result
            current_condition[f"{stage.name}_output"] = result
        
        output = {"result": result}
        if return_intermediates:
            output["intermediates"] = intermediates
        
        return output
    
    @staticmethod
    def from_config(config: str | dict | DictConfig) -> "MultiModalAssembler":
        """Build a multi-modal pipeline from config.
        
        Config format:
        ```yaml
        stages:
          - name: "text_to_image"
            model: {type: model/modular, ...}
            sampler: {type: engine/sampler, ...}
          - name: "upscale"
            model: {type: model/modular, ...}
            sampler: {type: engine/sampler, ...}
        ```
        """
        auto_discover()
        
        if isinstance(config, (str,)):
            config = OmegaConf.load(config)
        elif isinstance(config, dict):
            config = OmegaConf.create(config)
        
        assembler = MultiModalAssembler()
        
        for stage_cfg in config.get("stages", []):
            model = BlockBuilder.build(stage_cfg["model"])
            sampler_cfg = OmegaConf.to_container(stage_cfg.get("sampler", {}))
            sampler = DiffusionSampler(sampler_cfg, model=model)
            
            assembler.add_stage(
                name=stage_cfg.get("name", f"stage_{len(assembler.stages)}"),
                sampler=sampler,
            )
        
        return assembler
    
    def __len__(self):
        return len(self.stages)
    
    def __repr__(self):
        stages_str = " -> ".join(s.name for s in self.stages)
        return f"<MultiModalPipeline: {stages_str}>"
