"""Training panel component for Gradio UI."""
from __future__ import annotations

from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class TrainingStatus:
    """Current training status for UI display."""
    is_training: bool = False
    current_step: int = 0
    total_steps: int = 0
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = 0.0
    learning_rate: float = 0.0
    elapsed_seconds: float = 0.0
    loss_history: list = field(default_factory=list)
    
    @property
    def progress(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return min(1.0, self.current_step / self.total_steps)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_training": self.is_training,
            "step": f"{self.current_step}/{self.total_steps}",
            "epoch": f"{self.current_epoch}/{self.total_epochs}",
            "loss": f"{self.current_loss:.4f}",
            "lr": f"{self.learning_rate:.2e}",
            "progress": f"{self.progress * 100:.1f}%",
            "elapsed": f"{self.elapsed_seconds:.1f}s",
        }


def create_training_callback(status: TrainingStatus) -> Callable:
    """Create a callback that updates TrainingStatus from trainer."""
    def callback(trainer, loss_dict, step):
        status.is_training = True
        status.current_step = step
        status.current_loss = loss_dict["loss"].item()
        status.learning_rate = trainer.optimizer.param_groups[0]["lr"]
        status.current_epoch = trainer._epoch
        status.loss_history.append(status.current_loss)
    return callback
