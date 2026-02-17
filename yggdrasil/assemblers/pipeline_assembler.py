"""PipelineAssembler -- build complete generation/training pipelines from config."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from omegaconf import DictConfig, OmegaConf

from ..core.model.modular import ModularDiffusionModel
from ..core.engine.sampler import DiffusionSampler
from ..core.diffusion.process import AbstractDiffusionProcess
from ..core.block.builder import BlockBuilder
from ..core.block.registry import auto_discover


class PipelineAssembler:
    """Assembles complete pipelines (model + sampler + process + trainer).
    
    Reads a recipe YAML and constructs all components needed for
    either generation or training.
    """
    
    @staticmethod
    def from_config(
        config: str | dict | DictConfig,
    ) -> Dict[str, Any]:
        """Build a complete pipeline from config.
        
        Returns a dict with keys: model, sampler, process, (optionally) trainer_config.
        
        Args:
            config: Path to YAML, dict, or DictConfig
            
        Returns:
            Dict with assembled pipeline components
        """
        auto_discover()
        
        if isinstance(config, (str, Path)):
            config = OmegaConf.load(str(config))
        elif isinstance(config, dict):
            config = OmegaConf.create(config)
        
        result = {}
        
        # 1. Build model
        if "model" in config:
            model = BlockBuilder.build(config["model"])
            result["model"] = model
        
        # 2. Build diffusion process
        if "diffusion_process" in config:
            process = BlockBuilder.build(config["diffusion_process"])
            result["process"] = process
        
        # 3. Build sampler
        if "sampler" in config:
            sampler_config = OmegaConf.to_container(config["sampler"], resolve=True)
            sampler = DiffusionSampler(sampler_config, model=result.get("model"))
            
            # Attach process to sampler if not already in sampler config
            if "process" in result and getattr(sampler, "_process", None) is None:
                sampler._process = result["process"]
            
            result["sampler"] = sampler
        
        # 4. Training config (if present)
        if "training" in config:
            result["training_config"] = OmegaConf.to_container(config["training"])
        
        # 5. Loss config
        if "loss" in config:
            result["loss_config"] = OmegaConf.to_container(config["loss"])
        
        return result
    
    @staticmethod
    def from_recipe(recipe_name: str, **overrides) -> Dict[str, Any]:
        """Build pipeline from a named recipe.
        
        Args:
            recipe_name: Recipe name (e.g. "sd15_generate", "sd15_train_lora")
            
        Returns:
            Dict with assembled pipeline components
        """
        from ..configs import get_recipe
        config = get_recipe(recipe_name, **overrides)
        return PipelineAssembler.from_config(config)
    
    @staticmethod
    def for_generation(
        model: ModularDiffusionModel,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        solver: str = "ddim",
        process: str = "ddpm",
        **kwargs
    ) -> DiffusionSampler:
        """Quick assembly of a generation pipeline.
        
        Args:
            model: Pre-built model
            num_steps: Number of sampling steps
            guidance_scale: CFG scale
            solver: Solver type key
            process: Diffusion process type key
            
        Returns:
            Configured DiffusionSampler
        """
        auto_discover()
        
        sampler_config = {
            "type": "engine/sampler",
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
        }
        
        sampler = DiffusionSampler(sampler_config, model=model)
        
        # Build and attach process
        process_config = {"type": f"diffusion/process/{process}"}
        process_block = BlockBuilder.build(process_config)
        sampler._process = process_block
        
        solver_config = {"type": f"diffusion/solver/{solver}"}
        solver_block = BlockBuilder.build(solver_config)
        sampler._solver = solver_block
        
        return sampler
    
    @staticmethod
    def for_training(
        model: ModularDiffusionModel,
        process: Optional[AbstractDiffusionProcess] = None,
        loss_type: str = "epsilon",
        train_mode: str = "full",
        **training_kwargs
    ):
        """Quick assembly of a training pipeline.
        
        Args:
            model: Pre-built model
            process: Diffusion process (built if None)
            loss_type: Loss function type
            train_mode: "full", "adapter", or "finetune"
            **training_kwargs: TrainingConfig overrides
            
        Returns:
            Configured DiffusionTrainer
        """
        auto_discover()
        
        from ..training.trainer import DiffusionTrainer, TrainingConfig
        from ..training.loss import EpsilonLoss, VelocityLoss, FlowMatchingLoss
        
        # Build process if needed
        if process is None:
            process = BlockBuilder.build({"type": "diffusion/process/ddpm"})
        
        # Select loss
        loss_map = {
            "epsilon": EpsilonLoss,
            "velocity": VelocityLoss,
            "flow_matching": FlowMatchingLoss,
        }
        loss_cls = loss_map.get(loss_type, EpsilonLoss)
        loss_fn = loss_cls()
        
        # Build training config
        training_kwargs["train_mode"] = train_mode
        config = TrainingConfig.from_dict(training_kwargs)
        
        return DiffusionTrainer(
            model=model,
            process=process,
            loss_fn=loss_fn,
            config=config,
        )
