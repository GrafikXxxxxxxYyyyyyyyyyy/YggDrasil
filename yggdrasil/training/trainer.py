# yggdrasil/training/trainer.py
"""Универсальный тренер для диффузионных моделей любой модальности.

DiffusionTrainer работает с любой комбинацией:
    модель + процесс + loss + данные = обучение

Поддерживает:
    - Full model training
    - Adapter training (LoRA, DoRA)
    - Fine-tuning конкретных блоков
    - Mixed precision (fp16/bf16)
    - Gradient accumulation
    - EMA
    - Checkpoint save/resume
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
import time
import math

from ..core.model.modular import ModularDiffusionModel
from ..core.diffusion.process import AbstractDiffusionProcess
from .loss import DiffusionLoss, EpsilonLoss
from .data import AbstractDataSource


@dataclass
class TrainingConfig:
    """Конфигурация обучения."""
    # Основные параметры
    num_epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Gradient
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Mixed precision
    mixed_precision: str = "no"  # "no", "fp16", "bf16"
    
    # EMA
    use_ema: bool = False
    ema_decay: float = 0.9999
    
    # Optimizer
    optimizer: str = "adamw"  # "adamw", "adam", "sgd", "lion", "prodigy"
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # Scheduler
    lr_scheduler: str = "cosine"  # "constant", "cosine", "linear", "cosine_warmup"
    warmup_steps: int = 0
    
    # Checkpointing
    save_every: int = 500        # Сохранять каждые N шагов
    log_every: int = 10          # Логировать каждые N шагов
    sample_every: int = 200      # Генерировать примеры каждые N шагов
    checkpoint_dir: str = "checkpoints"
    
    # Training mode
    train_mode: str = "full"     # "full", "adapter", "finetune"
    trainable_blocks: Optional[List[str]] = None  # Какие блоки обучать (None = все)
    
    # Data
    num_workers: int = 0
    pin_memory: bool = True
    
    # Device
    device: str = "auto"         # "auto", "cuda", "mps", "cpu"
    
    @classmethod
    def from_dict(cls, d: dict) -> TrainingConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_yaml(cls, path: str) -> TrainingConfig:
        cfg = OmegaConf.load(path)
        return cls.from_dict(OmegaConf.to_container(cfg))


class EMAModel:
    """Exponential Moving Average для весов модели."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply(self, model: nn.Module):
        """Применить EMA веса к модели (для инференса)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        """Восстановить обычные веса."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class DiffusionTrainer:
    """Универсальный тренер диффузионных моделей.
    
    Работает с любой модальностью и любым процессом:
        trainer = DiffusionTrainer(model, process, loss, config)
        trainer.train(dataset)
    """
    
    def __init__(
        self,
        model: ModularDiffusionModel,
        process: AbstractDiffusionProcess,
        loss_fn: Optional[DiffusionLoss] = None,
        config: Optional[TrainingConfig | dict] = None,
    ):
        self.model = model
        self.process = process
        self.loss_fn = loss_fn or EpsilonLoss()
        self.config = config if isinstance(config, TrainingConfig) else TrainingConfig.from_dict(config or {})
        
        # Device
        self.device = self._resolve_device()
        
        # Callbacks
        self.callbacks: List[Callable] = []
        self._step_count = 0
        self._epoch = 0
        
        # Будут инициализированы в setup()
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
    
    def setup(self):
        """Подготовка к обучению."""
        # 1. Переносим на устройство
        self.model = self.model.to(self.device)
        self.process = self.process.to(self.device)
        
        # 2. Режим обучения
        self._setup_trainable_params()
        
        # 3. Optimizer
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("Нет обучаемых параметров! Проверьте train_mode и trainable_blocks.")
        
        self.optimizer = self._build_optimizer(trainable)
        
        # 4. Scheduler
        self.scheduler = self._build_scheduler()
        
        # 5. Mixed precision
        if self.config.mixed_precision == "fp16" and self.device.type == "cuda":
            self.scaler = GradScaler()
        
        # 6. EMA
        if self.config.use_ema:
            self.ema = EMAModel(self.model, self.config.ema_decay)
        
        # 7. Checkpoint dir
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        return self
    
    def _setup_trainable_params(self):
        """Настройка обучаемых параметров."""
        if self.config.train_mode == "full":
            self.model.train()
            # Замораживаем codec (VAE) если он есть — обычно не обучается
            nodes = getattr(self.model, "_graph", None)
            nodes = nodes.nodes if nodes and getattr(nodes, "nodes", None) else getattr(self.model, "_slot_children", {})
            if "codec" in nodes:
                for p in nodes["codec"].parameters():
                    p.requires_grad = False
        
        elif self.config.train_mode == "adapter":
            for p in self.model.parameters():
                p.requires_grad = False
            nodes = getattr(self.model, "_graph", None)
            nodes = nodes.nodes if nodes and getattr(nodes, "nodes", None) else getattr(self.model, "_slot_children", {})
            for adapter in (nodes.get("adapters") or []):
                for p in adapter.parameters():
                    p.requires_grad = True
        
        elif self.config.train_mode == "finetune":
            for p in self.model.parameters():
                p.requires_grad = False
            nodes = getattr(self.model, "_graph", None)
            nodes = nodes.nodes if nodes and getattr(nodes, "nodes", None) else getattr(self.model, "_slot_children", {})
            for block_name in (self.config.trainable_blocks or []):
                if block_name in nodes:
                    child = nodes[block_name]
                elif block_name == "adapters":
                    child = [nodes[k] for k in nodes if isinstance(k, str) and k.startswith("adapter_")]
                else:
                    child = None
                if child is not None:
                    for c in (child if isinstance(child, list) else [child]):
                        for p in c.parameters():
                            p.requires_grad = True
        
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Параметры: {trainable:,} обучаемых / {total:,} всего ({100*trainable/max(total,1):.1f}%)")
    
    def _build_optimizer(self, params) -> torch.optim.Optimizer:
        cfg = self.config
        if cfg.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=cfg.learning_rate, betas=cfg.betas,
                                      eps=cfg.eps, weight_decay=cfg.weight_decay)
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
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.num_epochs, eta_min=cfg.learning_rate * 0.01
            )
        elif cfg.lr_scheduler == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.01, total_iters=cfg.num_epochs
            )
        elif cfg.lr_scheduler == "cosine_warmup":
            def warmup_cosine(step):
                if step < cfg.warmup_steps:
                    return step / max(1, cfg.warmup_steps)
                progress = (step - cfg.warmup_steps) / max(1, cfg.num_epochs - cfg.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, warmup_cosine)
        return None
    
    # ==================== ОСНОВНОЙ ЦИКЛ ====================
    
    def train(self, dataset: AbstractDataSource) -> Dict[str, Any]:
        """Запустить обучение.
        
        Returns:
            Dict с историей обучения (losses, metrics)
        """
        self.setup()
        
        loader = dataset.get_dataloader(
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
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
                
                # Логирование
                if self._step_count % self.config.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - start_time
                    print(f"[Epoch {epoch+1}/{self.config.num_epochs}] "
                          f"Step {self._step_count} | Loss: {loss_value:.4f} | "
                          f"LR: {lr:.2e} | Time: {elapsed:.1f}s")
                    history["loss"].append(loss_value)
                    history["lr"].append(lr)
                    history["step"].append(self._step_count)
                
                # Checkpoint
                if self.config.save_every > 0 and self._step_count % self.config.save_every == 0 and self._step_count > 0:
                    self.save_checkpoint(f"step_{self._step_count}")
                
                # Callbacks
                for cb in self.callbacks:
                    cb(self, loss_dict, self._step_count)
            
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"=== Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f} ===")
            
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Финальный checkpoint
        self.save_checkpoint("final")
        
        total_time = time.time() - start_time
        print(f"Обучение завершено за {total_time:.1f}s ({self._step_count} шагов)")
        history["total_time"] = total_time
        return history
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Один шаг обучения.
        
        Работает с ЛЮБОЙ модальностью:
        1. Берём данные → encode → латенты
        2. Сэмплируем t, noise
        3. Forward process: x0 + noise → xt
        4. Модель предсказывает noise/velocity/x0
        5. Считаем loss
        """
        self.model.train()
        data = batch["data"].to(self.device)
        condition = batch.get("condition", None)
        
        # 1. Encode в латенты (если есть codec)
        with torch.no_grad():
            latents = self.model.encode(data)
        
        # 2. Сэмплируем таймстеп и шум
        batch_size = latents.shape[0]
        t = self._sample_timestep(batch_size)
        noise = torch.randn_like(latents)
        
        # 3. Forward diffusion: x0 → xt
        forward_result = self.process.forward_process(latents, t, noise)
        xt = forward_result["xt"]
        
        # Target зависит от параметризации
        target = forward_result.get("noise", forward_result.get("target", noise))
        
        # 4. Предсказание модели
        use_amp = self.config.mixed_precision == "fp16" and self.device.type == "cuda"
        
        if use_amp:
            with autocast():
                model_output = self.model(x=xt, t=t, condition=condition, return_dict=False)
                loss_dict = self.loss_fn.compute(model_output, target, t)
        else:
            model_output = self.model(x=xt, t=t, condition=condition, return_dict=False)
            loss_dict = self.loss_fn.compute(model_output, target, t)
        
        loss = loss_dict["loss"] / self.config.gradient_accumulation_steps
        
        # 5. Backward
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 6. Step (с gradient accumulation)
        self._step_count += 1
        if self._step_count % self.config.gradient_accumulation_steps == 0:
            if self.config.max_grad_norm > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.max_grad_norm
                )
            
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # EMA update
            if self.ema is not None:
                self.ema.update(self.model)
        
        return loss_dict
    
    def _sample_timestep(self, batch_size: int) -> torch.Tensor:
        """Сэмплируем таймстеп для обучения.
        
        Поддерживает разные стратегии семплирования:
        - uniform: равномерное распределение
        - logit_normal: для flow matching моделей (концентрация в середине)
        """
        num_train = getattr(self.process, "num_train_timesteps", 1000)
        
        if hasattr(self.process, "sample_training_timestep"):
            return self.process.sample_training_timestep(batch_size).to(self.device)
        
        # Дискретные таймстепы (DDPM-style)
        if num_train > 1:
            t = torch.randint(0, num_train, (batch_size,), device=self.device)
        else:
            # Непрерывные (Flow Matching)
            t = torch.rand(batch_size, device=self.device)
        
        return t
    
    # ==================== CHECKPOINTING ====================
    
    def save_checkpoint(self, name: str = "latest"):
        """Сохранить checkpoint."""
        path = Path(self.config.checkpoint_dir) / name
        path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "step": self._step_count,
            "epoch": self._epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        if self.ema is not None:
            checkpoint["ema_shadow"] = self.ema.shadow
        
        torch.save(checkpoint, path / "trainer_state.pt")
        OmegaConf.save(OmegaConf.structured(self.config), path / "training_config.yaml")
        print(f"Checkpoint сохранён: {path}")
    
    def load_checkpoint(self, name: str = "latest"):
        """Загрузить checkpoint."""
        path = Path(self.config.checkpoint_dir) / name
        checkpoint = torch.load(path / "trainer_state.pt", map_location=self.device, weights_only=False)
        
        self._step_count = checkpoint["step"]
        self._epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if self.ema is not None and "ema_shadow" in checkpoint:
            self.ema.shadow = checkpoint["ema_shadow"]
        
        print(f"Checkpoint загружен: {path} (step {self._step_count})")
    
    # ==================== CALLBACKS ====================
    
    def add_callback(self, fn: Callable):
        """Добавить callback: fn(trainer, loss_dict, step)"""
        self.callbacks.append(fn)
