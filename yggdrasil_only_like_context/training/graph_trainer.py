# yggdrasil/training/graph_trainer.py
"""GraphTrainer — train ANY subset of nodes in a ComputeGraph.

Key features:
- Works with ComputeGraph, not ModularDiffusionModel
- Per-node learning rates: train different parts at different speeds
- Training schedule: freeze/unfreeze nodes at specific epochs/steps
- Integrates with DatasetBlock and LossBlock as graph nodes
- Supports any combination of trainable nodes

Scenarios:
    - train_nodes=["backbone"] — full model training
    - train_nodes=["lora_adapter"] — LoRA training  
    - train_nodes=["controlnet"] — ControlNet training (backbone frozen)
    - train_nodes=["my_adapter"] — custom adapter
    - train_nodes=["text_encoder"] — text encoder fine-tuning
    - train_nodes=["vae_encoder", "vae_decoder"] — VAE training
    - train_nodes=["backbone", "my_adapter"] — joint training

Per-node LR example:
    trainer = GraphTrainer(
        graph=graph,
        train_nodes=["backbone", "text_encoder"],
        node_lr={"backbone": 1e-5, "text_encoder": 1e-6},
    )

Training schedule (freeze/unfreeze):
    trainer = GraphTrainer(
        graph=graph,
        train_nodes=["backbone", "adapter"],
        schedule=[
            {"epoch": 0, "freeze": ["backbone"]},        # First: train adapter only
            {"epoch": 10, "unfreeze": ["backbone"]},      # Then: joint training
            {"epoch": 10, "set_lr": {"backbone": 1e-6}},  # With lower LR for backbone
        ],
    )
"""
from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from omegaconf import DictConfig, OmegaConf

from yggdrasil.core.graph.graph import ComputeGraph
from yggdrasil.core.graph.executor import GraphExecutor

logger = logging.getLogger(__name__)


@dataclass
class GraphTrainingConfig:
    """Training configuration for ComputeGraph."""
    # Core
    num_epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Per-node learning rates: {"node_name": lr}
    node_lr: Optional[Dict[str, float]] = None
    
    # Training schedule: list of actions at specific epochs/steps
    # Each entry: {"epoch": N, "freeze": [...], "unfreeze": [...], "set_lr": {"node": lr}}
    # Or: {"step": N, "freeze": [...], "unfreeze": [...], "set_lr": {"node": lr}}
    schedule: Optional[List[Dict[str, Any]]] = None
    
    # Gradient
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Mixed precision
    mixed_precision: str = "no"  # "no", "fp16", "bf16"
    
    # EMA
    use_ema: bool = False
    ema_decay: float = 0.9999
    
    # Optimizer
    optimizer: str = "adamw"  # "adamw", "adam", "sgd", "adam8bit"
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # Scheduler
    lr_scheduler: str = "cosine"  # "constant", "cosine", "cosine_warmup", "linear"
    warmup_steps: int = 0
    
    # Checkpointing
    save_every: int = 500
    log_every: int = 10
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "auto"
    
    # Loss
    loss_type: str = "epsilon"  # "epsilon", "velocity", "flow_matching", "x0"
    loss_output_name: str = "noise_pred"  # Graph output to use as prediction
    
    # Validation
    val_every: int = 0  # Validate every N steps (0 = disabled)
    
    @classmethod
    def from_dict(cls, d: dict) -> GraphTrainingConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_yaml(cls, path: str) -> GraphTrainingConfig:
        cfg = OmegaConf.load(path)
        return cls.from_dict(OmegaConf.to_container(cfg))


class GraphTrainer:
    """Train any subset of nodes in a ComputeGraph.
    
    Features:
    - Per-node learning rates for fine-grained control
    - Training schedule: freeze/unfreeze nodes at specific epochs/steps
    - DatasetBlock + LossBlock integration as graph nodes
    - EMA, mixed precision, gradient accumulation
    - Checkpoint save/resume (only trained node weights)
    
    Basic usage::
    
        graph = ComputeGraph.from_template("sd15_txt2img")
        trainer = GraphTrainer(
            graph=graph,
            train_nodes=["backbone"],
            config=GraphTrainingConfig(learning_rate=1e-4),
        )
        trainer.train(dataset)
    
    Per-node LR::
    
        trainer = GraphTrainer(
            graph=graph,
            train_nodes=["backbone", "text_encoder"],
            config=GraphTrainingConfig(
                node_lr={"backbone": 1e-5, "text_encoder": 1e-6}
            ),
        )
    
    Training schedule (progressive unfreezing)::
    
        trainer = GraphTrainer(
            graph=graph,
            train_nodes=["backbone", "adapter"],
            config=GraphTrainingConfig(
                schedule=[
                    {"epoch": 0, "freeze": ["backbone"]},
                    {"epoch": 10, "unfreeze": ["backbone"]},
                    {"epoch": 10, "set_lr": {"backbone": 1e-6}},
                ]
            ),
        )
    """
    
    def __init__(
        self,
        graph: ComputeGraph,
        train_nodes: Optional[List[str]] = None,
        freeze_nodes: Optional[List[str]] = None,
        config: Optional[GraphTrainingConfig | dict] = None,
        loss_fn: Optional[Callable] = None,
        val_graph: Optional[ComputeGraph] = None,
    ):
        self.graph = graph
        # T2: either train_nodes (train only these) or freeze_nodes (train all except these)
        if freeze_nodes is not None:
            if train_nodes is not None:
                raise ValueError("Provide either train_nodes or freeze_nodes, not both.")
            frozen_set = set(freeze_nodes)
            self.train_nodes = [n for n in graph.nodes if n not in frozen_set]
        else:
            self.train_nodes = list(train_nodes or list(graph.nodes))
        self.config = (
            config if isinstance(config, GraphTrainingConfig)
            else GraphTrainingConfig.from_dict(config or {})
        )
        self.loss_fn = loss_fn or self._default_loss_fn
        self.val_graph = val_graph
        
        # Merge node_lr from config
        self.node_lr: Dict[str, float] = dict(self.config.node_lr or {})
        
        # Training schedule
        self.schedule: List[Dict[str, Any]] = list(self.config.schedule or [])
        self._applied_schedule_epochs: set = set()
        self._applied_schedule_steps: set = set()
        
        # Runtime state
        self.device = self._resolve_device()
        self.executor = GraphExecutor(no_grad=False)
        self.callbacks: List[Callable] = []
        self._step_count = 0
        self._epoch = 0
        
        # Active/frozen tracking
        self._frozen_nodes: set = set()
        
        # Will be set in setup()
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.ema = None
        
        # Node-to-param-group mapping for per-node LR
        self._node_param_groups: Dict[str, int] = {}
    
    def _resolve_device(self) -> torch.device:
        if self.config.device != "auto":
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def setup(self) -> GraphTrainer:
        """Prepare for training."""
        # 1. Move all nodes to device
        for name, block in self.graph.nodes.items():
            if hasattr(block, 'to'):
                block.to(self.device)
        
        # 2. Freeze/unfreeze
        self._setup_trainable_params()
        
        # 3. Apply initial schedule actions (epoch=0)
        self._apply_schedule(epoch=0, step=0)
        
        # 4. Build optimizer with per-node param groups
        self.optimizer = self._build_optimizer_with_groups()
        if self.optimizer is None:
            raise ValueError(
                f"No trainable parameters found in nodes: {self.train_nodes}. "
                f"Available nodes: {list(self.graph.nodes.keys())}"
            )
        
        # 5. Scheduler
        self.scheduler = self._build_scheduler()
        
        # 6. Mixed precision
        if self.config.mixed_precision == "fp16" and self.device.type == "cuda":
            self.scaler = GradScaler()
        
        # 7. EMA
        if self.config.use_ema:
            self.ema = _GraphEMA(self.graph, self.train_nodes, self.config.ema_decay)
        
        # 8. Checkpoint dir
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        return self
    
    def _setup_trainable_params(self):
        """Freeze everything, then unfreeze only train_nodes."""
        # Freeze ALL
        for name, block in self.graph.nodes.items():
            if hasattr(block, 'parameters'):
                for p in block.parameters():
                    p.requires_grad = False
        
        # Unfreeze train_nodes (supports stage/inner_node for combined pipeline)
        total_params = 0
        trainable_params = 0
        for name in self.train_nodes:
            if name in self._frozen_nodes:
                continue
            block, _ = self._resolve_trainable_block(name)
            if block is None:
                logger.warning(f"Train node '{name}' not found in graph. Skipping.")
                continue
            if hasattr(block, 'parameters'):
                for p in block.parameters():
                    p.requires_grad = True
                    trainable_params += p.numel()
            if hasattr(block, 'train'):
                block.train()
        
        for name, block in self.graph.nodes.items():
            if hasattr(block, 'parameters'):
                total_params += sum(p.numel() for p in block.parameters())
        
        logger.info(
            f"Training nodes: {self.train_nodes} | "
            f"Frozen: {self._frozen_nodes} | "
            f"Trainable: {trainable_params:,} / {total_params:,} "
            f"({100*trainable_params/max(total_params,1):.1f}%)"
        )
    
    def _build_optimizer_with_groups(self):
        """Build optimizer with per-node parameter groups for different LRs."""
        param_groups = []
        self._node_param_groups = {}
        
        for name in self.train_nodes:
            if name in self._frozen_nodes:
                continue
            block, _ = self._resolve_trainable_block(name)
            if block is None or not hasattr(block, 'parameters'):
                continue
            
            params = [p for p in block.parameters() if p.requires_grad]
            if not params:
                continue
            
            lr = self.node_lr.get(name, self.config.learning_rate)
            group_idx = len(param_groups)
            param_groups.append({
                "params": params,
                "lr": lr,
                "name": name,
            })
            self._node_param_groups[name] = group_idx
        
        if not param_groups:
            return None
        
        cfg = self.config
        if cfg.optimizer == "adamw":
            return torch.optim.AdamW(
                param_groups, lr=cfg.learning_rate, betas=cfg.betas,
                eps=cfg.eps, weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "adam":
            return torch.optim.Adam(
                param_groups, lr=cfg.learning_rate, betas=cfg.betas, eps=cfg.eps
            )
        elif cfg.optimizer == "sgd":
            return torch.optim.SGD(
                param_groups, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    
    def _build_scheduler(self):
        cfg = self.config
        if cfg.lr_scheduler == "constant":
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
        elif cfg.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.num_epochs, eta_min=cfg.learning_rate * 0.01
            )
        elif cfg.lr_scheduler == "cosine_warmup":
            def warmup_cosine(step):
                if step < cfg.warmup_steps:
                    return step / max(1, cfg.warmup_steps)
                progress = (step - cfg.warmup_steps) / max(1, cfg.num_epochs - cfg.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, warmup_cosine)
        elif cfg.lr_scheduler == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.01, total_iters=cfg.num_epochs
            )
        return None
    
    # ==================== SCHEDULE ====================
    
    def _apply_schedule(self, epoch: int, step: int):
        """Apply training schedule actions for the current epoch/step."""
        for action in self.schedule:
            trigger_epoch = action.get("epoch")
            trigger_step = action.get("step")
            
            # Check epoch triggers
            if trigger_epoch is not None:
                action_id = f"epoch_{trigger_epoch}_{id(action)}"
                if epoch >= trigger_epoch and action_id not in self._applied_schedule_epochs:
                    self._execute_schedule_action(action)
                    self._applied_schedule_epochs.add(action_id)
            
            # Check step triggers
            if trigger_step is not None:
                action_id = f"step_{trigger_step}_{id(action)}"
                if step >= trigger_step and action_id not in self._applied_schedule_steps:
                    self._execute_schedule_action(action)
                    self._applied_schedule_steps.add(action_id)
    
    def _execute_schedule_action(self, action: Dict[str, Any]):
        """Execute a single schedule action."""
        # Freeze nodes
        for node_name in action.get("freeze", []):
            self._freeze_node(node_name)
            logger.info(f"Schedule: froze {node_name}")
        
        # Unfreeze nodes
        for node_name in action.get("unfreeze", []):
            self._unfreeze_node(node_name)
            logger.info(f"Schedule: unfroze {node_name}")
        
        # Set learning rates
        for node_name, lr in action.get("set_lr", {}).items():
            self._set_node_lr(node_name, lr)
            logger.info(f"Schedule: set LR for {node_name} to {lr}")
    
    def _freeze_node(self, name: str):
        """Freeze a specific node (supports stage/inner_node)."""
        self._frozen_nodes.add(name)
        block, _ = self._resolve_trainable_block(name)
        if block is not None and hasattr(block, 'parameters'):
            for p in block.parameters():
                p.requires_grad = False
            if hasattr(block, 'eval'):
                block.eval()
    
    def _unfreeze_node(self, name: str):
        """Unfreeze a specific node (supports stage/inner_node)."""
        self._frozen_nodes.discard(name)
        block, _ = self._resolve_trainable_block(name)
        if block is not None and hasattr(block, 'parameters'):
            for p in block.parameters():
                p.requires_grad = True
            if hasattr(block, 'train'):
                block.train()
            # Add to optimizer if not already present
            if self.optimizer is not None and name not in self._node_param_groups:
                params = [p for p in block.parameters() if p.requires_grad]
                if params:
                    lr = self.node_lr.get(name, self.config.learning_rate)
                    self.optimizer.add_param_group({
                        "params": params, "lr": lr, "name": name,
                    })
                    self._node_param_groups[name] = len(self.optimizer.param_groups) - 1
    
    def _set_node_lr(self, name: str, lr: float):
        """Set learning rate for a specific node."""
        self.node_lr[name] = lr
        if self.optimizer is not None and name in self._node_param_groups:
            group_idx = self._node_param_groups[name]
            self.optimizer.param_groups[group_idx]["lr"] = lr
    
    def get_node_lr(self, name: str) -> Optional[float]:
        """Get current learning rate for a node."""
        if self.optimizer is not None and name in self._node_param_groups:
            group_idx = self._node_param_groups[name]
            return self.optimizer.param_groups[group_idx]["lr"]
        return self.node_lr.get(name, self.config.learning_rate)
    
    # ==================== TRAINING ====================
    
    def train(self, dataset, val_dataset=None, **kwargs) -> Dict[str, Any]:
        """Run training loop.
        
        Args:
            dataset: Must provide .get_dataloader() method.
            val_dataset: Optional validation dataset.
        """
        self.setup()
        
        loader = dataset.get_dataloader(
            batch_size=self.config.batch_size, shuffle=True,
            num_workers=kwargs.get("num_workers", 0),
            pin_memory=kwargs.get("pin_memory", True),
        )
        
        history = {"loss": [], "lr": {}, "step": [], "val_loss": []}
        for name in self.train_nodes:
            history["lr"][name] = []
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self._epoch = epoch
            
            # Apply schedule for this epoch
            self._apply_schedule(epoch=epoch, step=self._step_count)
            
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(loader):
                # Apply step-based schedule
                self._apply_schedule(epoch=epoch, step=self._step_count)
                
                loss_dict = self.train_step(batch)
                loss_value = loss_dict["loss"].item()
                epoch_loss += loss_value
                num_batches += 1
                
                if self._step_count % self.config.log_every == 0:
                    lr_info = {
                        name: self.get_node_lr(name) 
                        for name in self.train_nodes
                        if name not in self._frozen_nodes
                    }
                    logger.info(
                        f"[Epoch {epoch+1}/{self.config.num_epochs}] "
                        f"Step {self._step_count} | Loss: {loss_value:.4f} | "
                        f"LR: {lr_info}"
                    )
                    history["loss"].append(loss_value)
                    for name, lr_val in lr_info.items():
                        history["lr"].setdefault(name, []).append(lr_val)
                    history["step"].append(self._step_count)
                
                # Validation
                if (self.config.val_every > 0 and 
                    self._step_count % self.config.val_every == 0 and 
                    val_dataset is not None):
                    val_loss = self.validate(val_dataset)
                    history["val_loss"].append(val_loss)
                    logger.info(f"Validation loss: {val_loss:.4f}")
                
                if self.config.save_every > 0 and self._step_count % self.config.save_every == 0 and self._step_count > 0:
                    self.save_checkpoint(f"step_{self._step_count}")
                
                for cb in self.callbacks:
                    cb(self, loss_dict, self._step_count)
            
            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"=== Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} ===")
            
            if self.scheduler is not None:
                self.scheduler.step()
        
        self.save_checkpoint("final")
        history["total_time"] = time.time() - start_time
        return history
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """One training step through the full graph.
        
        1. Prepare inputs from batch
        2. Forward through entire graph (with gradients for train_nodes)
        3. Compute loss on graph outputs
        4. Backward through train_nodes only
        5. Optimizer step
        """
        # Prepare inputs
        graph_inputs = self._prepare_inputs(batch)
        
        # Forward through graph
        use_amp = self.config.mixed_precision == "fp16" and self.device.type == "cuda"
        
        if use_amp:
            with autocast():
                outputs = self.executor.execute_training(self.graph, **graph_inputs)
        else:
            outputs = self.executor.execute_training(self.graph, **graph_inputs)
        
        # T3: if graph has a loss node output, use it for backward; else use loss_fn
        if "loss" in outputs and isinstance(outputs.get("loss"), torch.Tensor):
            loss_dict = {"loss": outputs["loss"]}
        else:
            loss_dict = self.loss_fn(outputs, batch, self.graph)
        
        loss = loss_dict["loss"] / self.config.gradient_accumulation_steps
        
        # Backward
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Step
        self._step_count += 1
        if self._step_count % self.config.gradient_accumulation_steps == 0:
            if self.config.max_grad_norm > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                params = self._get_all_trainable_params()
                nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
            
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            if self.ema is not None:
                self.ema.update()
        
        return loss_dict
    
    def validate(self, val_dataset) -> float:
        """Run validation on a dataset."""
        was_training = {}
        for name in self.train_nodes:
            block, _ = self._resolve_trainable_block(name)
            if block is not None and hasattr(block, 'training'):
                was_training[name] = block.training
                if hasattr(block, 'eval'):
                    block.eval()
        
        loader = val_dataset.get_dataloader(batch_size=self.config.batch_size, shuffle=False)
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                graph_inputs = self._prepare_inputs(batch)
                outputs = self.executor.execute(self.graph, **graph_inputs)
                if "loss" in outputs and isinstance(outputs.get("loss"), torch.Tensor):
                    total_loss += outputs["loss"].item()
                else:
                    loss_dict = self.loss_fn(outputs, batch, self.graph)
                    total_loss += loss_dict["loss"].item()
                num_batches += 1
        
        # Restore training mode
        for name, was_train in was_training.items():
            block, _ = self._resolve_trainable_block(name)
            if block is not None and was_train and hasattr(block, 'train'):
                block.train()
        
        return total_loss / max(num_batches, 1)
    
    def _prepare_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Convert batch to graph inputs."""
        inputs = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.device)
            else:
                inputs[key] = value
        return inputs
    
    def _resolve_trainable_block(self, name: str):
        """Resolve train node name to (block, checkpoint_key). Supports stage/inner (e.g. stage0/lora_adapter)."""
        if "/" in name:
            stage_name, inner_name = name.split("/", 1)
            stage_name = stage_name.strip()
            inner_name = inner_name.strip()
            block = self.graph.nodes.get(stage_name)
            if block is not None and hasattr(block, "graph") and hasattr(block.graph, "nodes"):
                inner = block.graph.nodes.get(inner_name)
                if inner is not None:
                    return inner, name
            return None, name
        block = self.graph.nodes.get(name)
        return block, name

    def _get_all_trainable_params(self) -> List[torch.nn.Parameter]:
        """Collect all trainable parameters from train_nodes (supports stage/inner_node)."""
        params = []
        for name in self.train_nodes:
            if name in self._frozen_nodes:
                continue
            block, _ = self._resolve_trainable_block(name)
            if block is not None and hasattr(block, 'parameters'):
                params.extend(p for p in block.parameters() if p.requires_grad)
        return params
    
    def _default_loss_fn(self, outputs, batch, graph):
        """Default loss: MSE between prediction and target."""
        prediction = outputs.get(self.config.loss_output_name, outputs.get("output"))
        target = batch.get("target", batch.get("noise"))
        
        if prediction is None or target is None:
            raise ValueError("Cannot compute loss: prediction or target is None")
        
        if isinstance(target, torch.Tensor):
            target = target.to(prediction.device)
        
        loss = torch.nn.functional.mse_loss(prediction, target)
        return {"loss": loss}
    
    # ==================== CHECKPOINTING ====================
    
    def save_checkpoint(self, name: str = "latest"):
        """Save checkpoint with trained node weights and training state."""
        path = Path(self.config.checkpoint_dir) / name
        path.mkdir(parents=True, exist_ok=True)
        
        # Save only train_nodes weights (supports stage/inner_node)
        state = {}
        for node_name in self.train_nodes:
            block, key = self._resolve_trainable_block(node_name)
            if block is not None and hasattr(block, 'state_dict'):
                state[key] = block.state_dict()
        
        checkpoint = {
            "step": self._step_count,
            "epoch": self._epoch,
            "train_nodes": self.train_nodes,
            "train_nodes_state": state,
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "node_lr": self.node_lr,
            "frozen_nodes": list(self._frozen_nodes),
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path / "graph_trainer_state.pt")
        
        # Save graph structure
        if hasattr(self.graph, 'to_yaml'):
            self.graph.to_yaml(path / "graph.yaml")
        
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, name: str = "latest"):
        """Load checkpoint."""
        path = Path(self.config.checkpoint_dir) / name
        checkpoint = torch.load(
            path / "graph_trainer_state.pt", 
            map_location=self.device, weights_only=False
        )
        
        self._step_count = checkpoint["step"]
        self._epoch = checkpoint["epoch"]
        self._frozen_nodes = set(checkpoint.get("frozen_nodes", []))
        self.node_lr = checkpoint.get("node_lr", {})
        
        for node_name, state in checkpoint.get("train_nodes_state", {}).items():
            block, _ = self._resolve_trainable_block(node_name)
            if block is not None and hasattr(block, 'load_state_dict'):
                block.load_state_dict(state)
        
        logger.info(f"Checkpoint loaded: {path} (step {self._step_count})")
    
    def save_trained_nodes(self, path: str):
        """Export only the trained node weights (e.g., for sharing LoRA weights)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        for node_name in self.train_nodes:
            block, _ = self._resolve_trainable_block(node_name)
            if block is not None and hasattr(block, 'state_dict'):
                safe_name = node_name.replace("/", "_")
                torch.save(block.state_dict(), path / f"{safe_name}.pt")
        
        logger.info(f"Trained nodes exported: {path}")
    
    def load_trained_nodes(self, path: str):
        """Import trained node weights (supports stage/inner_node)."""
        path = Path(path)
        for node_name in self.train_nodes:
            safe_name = node_name.replace("/", "_")
            weight_file = path / f"{safe_name}.pt"
            if weight_file.exists():
                block, _ = self._resolve_trainable_block(node_name)
                if block is not None and hasattr(block, 'load_state_dict'):
                    state = torch.load(weight_file, map_location=self.device, weights_only=True)
                    block.load_state_dict(state)
                    logger.info(f"Loaded weights for {node_name} from {weight_file}")
    
    def add_callback(self, fn: Callable):
        """Add training callback: fn(trainer, loss_dict, step)"""
        self.callbacks.append(fn)
    
    def summary(self) -> str:
        """Return a summary of training configuration."""
        lines = [
            "=== GraphTrainer Summary ===",
            f"Device: {self.device}",
            f"Train nodes: {self.train_nodes}",
            f"Frozen nodes: {self._frozen_nodes}",
        ]
        for name in self.train_nodes:
            block, _ = self._resolve_trainable_block(name)
            if block is not None and hasattr(block, 'parameters'):
                n_params = sum(p.numel() for p in block.parameters())
                n_train = sum(p.numel() for p in block.parameters() if p.requires_grad)
                lr = self.get_node_lr(name)
                lines.append(
                    f"  {name}: {n_train:,}/{n_params:,} trainable, lr={lr}"
                )
        lines.append(f"Optimizer: {self.config.optimizer}")
        lines.append(f"Scheduler: {self.config.lr_scheduler}")
        lines.append(f"Schedule actions: {len(self.schedule)}")
        return "\n".join(lines)


def _resolve_block_for_ema(graph: ComputeGraph, name: str):
    """Resolve train node name to block (for EMA). Supports stage/inner_node."""
    if "/" in name:
        stage_name, inner_name = name.split("/", 1)
        stage_name, inner_name = stage_name.strip(), inner_name.strip()
        block = graph.nodes.get(stage_name)
        if block is not None and hasattr(block, "graph") and hasattr(block.graph, "nodes"):
            return block.graph.nodes.get(inner_name)
        return None
    return graph.nodes.get(name)


class _GraphEMA:
    """EMA for specific nodes in a ComputeGraph (supports stage/inner_node)."""

    def __init__(self, graph: ComputeGraph, train_nodes: List[str], decay: float = 0.9999):
        self.graph = graph
        self.train_nodes = train_nodes
        self.decay = decay
        self.shadow = {}

        for node_name in train_nodes:
            block = _resolve_block_for_ema(graph, node_name)
            if block is not None and hasattr(block, 'named_parameters'):
                for pname, param in block.named_parameters():
                    if param.requires_grad:
                        key = f"{node_name}.{pname}"
                        self.shadow[key] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for node_name in self.train_nodes:
            block = _resolve_block_for_ema(self.graph, node_name)
            if block is not None and hasattr(block, 'named_parameters'):
                for pname, param in block.named_parameters():
                    key = f"{node_name}.{pname}"
                    if key in self.shadow:
                        self.shadow[key] = self.decay * self.shadow[key] + (1 - self.decay) * param.data

    def apply(self):
        self._backup = {}
        for node_name in self.train_nodes:
            block = _resolve_block_for_ema(self.graph, node_name)
            if block is not None and hasattr(block, 'named_parameters'):
                for pname, param in block.named_parameters():
                    key = f"{node_name}.{pname}"
                    if key in self.shadow:
                        self._backup[key] = param.data.clone()
                        param.data = self.shadow[key]

    def restore(self):
        for node_name in self.train_nodes:
            block = _resolve_block_for_ema(self.graph, node_name)
            if block is not None and hasattr(block, 'named_parameters'):
                for pname, param in block.named_parameters():
                    key = f"{node_name}.{pname}"
                    if key in self._backup:
                        param.data = self._backup[key]
        self._backup = {}
