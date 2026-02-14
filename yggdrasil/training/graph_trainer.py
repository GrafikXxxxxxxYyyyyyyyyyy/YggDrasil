# yggdrasil/training/graph_trainer.py
"""GraphTrainer — обучение ЛЮБОГО подмножества узлов в ComputeGraph.

Ключевое отличие от DiffusionTrainer:
- Работает с ComputeGraph, а не с ModularDiffusionModel
- Позволяет указать train_nodes — какие узлы графа обучать
- Остальные узлы заморожены
- Forward проходит через ВЕСЬ граф
- Backward считается только через train_nodes

Сценарии:
    - train_nodes=["backbone"] — full model training
    - train_nodes=["lora_adapter"] — LoRA training
    - train_nodes=["controlnet"] — ControlNet training (backbone frozen)
    - train_nodes=["my_adapter"] — custom adapter
    - train_nodes=["text_encoder"] — text encoder fine-tuning
    - train_nodes=["vae_encoder", "vae_decoder"] — VAE training
    - train_nodes=["backbone", "my_adapter"] — joint training
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
    """Конфигурация обучения графа."""
    # Основные
    num_epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Gradient
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Mixed precision
    mixed_precision: str = "no"
    
    # EMA
    use_ema: bool = False
    ema_decay: float = 0.9999
    
    # Optimizer
    optimizer: str = "adamw"
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # Scheduler
    lr_scheduler: str = "cosine"
    warmup_steps: int = 0
    
    # Checkpointing
    save_every: int = 500
    log_every: int = 10
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "auto"
    
    # Loss
    loss_type: str = "epsilon"  # "epsilon", "velocity", "flow_matching", "x0"
    loss_output_name: str = "noise_pred"  # Name of the graph output to use as prediction
    
    @classmethod
    def from_dict(cls, d: dict) -> GraphTrainingConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class GraphTrainer:
    """Обучение любого подмножества узлов ComputeGraph.
    
    Использование::
    
        graph = ComputeGraph.from_template("sd15_txt2img")
        graph.replace_node("backbone", MyCustomBackbone(config))
        
        trainer = GraphTrainer(
            graph=graph,
            train_nodes=["backbone"],
            config=GraphTrainingConfig(learning_rate=1e-4, num_epochs=100),
        )
        trainer.train(dataset)
    """
    
    def __init__(
        self,
        graph: ComputeGraph,
        train_nodes: List[str],
        config: Optional[GraphTrainingConfig | dict] = None,
        loss_fn: Optional[Callable] = None,
    ):
        self.graph = graph
        self.train_nodes = train_nodes
        self.config = (
            config if isinstance(config, GraphTrainingConfig)
            else GraphTrainingConfig.from_dict(config or {})
        )
        self.loss_fn = loss_fn or self._default_loss_fn
        
        # Runtime state
        self.device = self._resolve_device()
        self.executor = GraphExecutor(no_grad=False)
        self.callbacks: List[Callable] = []
        self._step_count = 0
        self._epoch = 0
        
        # Will be set in setup()
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.ema = None
    
    def _resolve_device(self) -> torch.device:
        if self.config.device != "auto":
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def setup(self) -> GraphTrainer:
        """Подготовка к обучению."""
        # 1. Move all nodes to device
        for name, block in self.graph.nodes.items():
            if hasattr(block, 'to'):
                block.to(self.device)
        
        # 2. Freeze/unfreeze
        self._setup_trainable_params()
        
        # 3. Collect trainable parameters
        trainable_params = self._get_trainable_params()
        if not trainable_params:
            raise ValueError(
                f"No trainable parameters found in nodes: {self.train_nodes}. "
                f"Available nodes: {list(self.graph.nodes.keys())}"
            )
        
        # 4. Optimizer
        self.optimizer = self._build_optimizer(trainable_params)
        
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
        
        # Unfreeze train_nodes
        total_params = 0
        trainable_params = 0
        for name in self.train_nodes:
            if name not in self.graph.nodes:
                logger.warning(f"Train node '{name}' not found in graph. Skipping.")
                continue
            block = self.graph.nodes[name]
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
            f"Trainable: {trainable_params:,} / {total_params:,} "
            f"({100*trainable_params/max(total_params,1):.1f}%)"
        )
    
    def _get_trainable_params(self) -> List[torch.nn.Parameter]:
        """Collect all trainable parameters from train_nodes."""
        params = []
        for name in self.train_nodes:
            block = self.graph.nodes.get(name)
            if block is not None and hasattr(block, 'parameters'):
                params.extend(p for p in block.parameters() if p.requires_grad)
        return params
    
    def _build_optimizer(self, params):
        cfg = self.config
        if cfg.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=cfg.learning_rate, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "adam":
            return torch.optim.Adam(params, lr=cfg.learning_rate, betas=cfg.betas, eps=cfg.eps)
        elif cfg.optimizer == "sgd":
            return torch.optim.SGD(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    
    def _build_scheduler(self):
        cfg = self.config
        if cfg.lr_scheduler == "constant":
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
        elif cfg.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.num_epochs, eta_min=cfg.learning_rate * 0.01)
        elif cfg.lr_scheduler == "cosine_warmup":
            def warmup_cosine(step):
                if step < cfg.warmup_steps:
                    return step / max(1, cfg.warmup_steps)
                progress = (step - cfg.warmup_steps) / max(1, cfg.num_epochs - cfg.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, warmup_cosine)
        return None
    
    # ==================== TRAINING ====================
    
    def train(self, dataset, **kwargs) -> Dict[str, Any]:
        """Run training loop.
        
        Args:
            dataset: Must provide .get_dataloader() method.
        """
        self.setup()
        
        loader = dataset.get_dataloader(
            batch_size=self.config.batch_size, shuffle=True,
            num_workers=kwargs.get("num_workers", 0),
            pin_memory=kwargs.get("pin_memory", True),
        )
        
        history = {"loss": [], "lr": [], "step": []}
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self._epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(loader):
                loss_dict = self.train_step(batch)
                loss_value = loss_dict["loss"].item()
                epoch_loss += loss_value
                num_batches += 1
                
                if self._step_count % self.config.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"[Epoch {epoch+1}/{self.config.num_epochs}] "
                        f"Step {self._step_count} | Loss: {loss_value:.4f} | LR: {lr:.2e}"
                    )
                    history["loss"].append(loss_value)
                    history["lr"].append(lr)
                    history["step"].append(self._step_count)
                
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
                loss_dict = self.loss_fn(outputs, batch, self.graph)
        else:
            outputs = self.executor.execute_training(self.graph, **graph_inputs)
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
                params = self._get_trainable_params()
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
    
    def _prepare_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Convert batch to graph inputs."""
        inputs = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.device)
            else:
                inputs[key] = value
        return inputs
    
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
        path = Path(self.config.checkpoint_dir) / name
        path.mkdir(parents=True, exist_ok=True)
        
        # Save only train_nodes weights
        state = {}
        for node_name in self.train_nodes:
            block = self.graph.nodes.get(node_name)
            if block is not None and hasattr(block, 'state_dict'):
                state[node_name] = block.state_dict()
        
        checkpoint = {
            "step": self._step_count,
            "epoch": self._epoch,
            "train_nodes_state": state,
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path / "graph_trainer_state.pt")
        
        # Save graph structure
        self.graph.to_yaml(path / "graph.yaml")
        
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, name: str = "latest"):
        path = Path(self.config.checkpoint_dir) / name
        checkpoint = torch.load(path / "graph_trainer_state.pt", map_location=self.device, weights_only=False)
        
        self._step_count = checkpoint["step"]
        self._epoch = checkpoint["epoch"]
        
        for node_name, state in checkpoint.get("train_nodes_state", {}).items():
            block = self.graph.nodes.get(node_name)
            if block is not None and hasattr(block, 'load_state_dict'):
                block.load_state_dict(state)
        
        logger.info(f"Checkpoint loaded: {path} (step {self._step_count})")
    
    def add_callback(self, fn: Callable):
        self.callbacks.append(fn)


class _GraphEMA:
    """EMA for specific nodes in a ComputeGraph."""
    
    def __init__(self, graph: ComputeGraph, train_nodes: List[str], decay: float = 0.9999):
        self.graph = graph
        self.train_nodes = train_nodes
        self.decay = decay
        self.shadow = {}
        
        for node_name in train_nodes:
            block = graph.nodes.get(node_name)
            if block is not None and hasattr(block, 'named_parameters'):
                for pname, param in block.named_parameters():
                    if param.requires_grad:
                        key = f"{node_name}.{pname}"
                        self.shadow[key] = param.data.clone()
    
    @torch.no_grad()
    def update(self):
        for node_name in self.train_nodes:
            block = self.graph.nodes.get(node_name)
            if block is not None and hasattr(block, 'named_parameters'):
                for pname, param in block.named_parameters():
                    key = f"{node_name}.{pname}"
                    if key in self.shadow:
                        self.shadow[key] = self.decay * self.shadow[key] + (1 - self.decay) * param.data
    
    def apply(self):
        self._backup = {}
        for node_name in self.train_nodes:
            block = self.graph.nodes.get(node_name)
            if block is not None and hasattr(block, 'named_parameters'):
                for pname, param in block.named_parameters():
                    key = f"{node_name}.{pname}"
                    if key in self.shadow:
                        self._backup[key] = param.data.clone()
                        param.data = self.shadow[key]
    
    def restore(self):
        for node_name in self.train_nodes:
            block = self.graph.nodes.get(node_name)
            if block is not None and hasattr(block, 'named_parameters'):
                for pname, param in block.named_parameters():
                    key = f"{node_name}.{pname}"
                    if key in self._backup:
                        param.data = self._backup[key]
        self._backup = {}
