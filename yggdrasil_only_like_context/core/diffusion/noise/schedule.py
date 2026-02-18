import torch
import math
from abc import abstractmethod
from typing import Optional
from omegaconf import DictConfig

from ....core.block.base import AbstractBaseBlock
from ....core.block.registry import register_block
from ....core.block.port import Port, InputPort, OutputPort, TensorSpec


@register_block("noise/schedule/abstract")
class NoiseSchedule(AbstractBaseBlock):
    """Abstract noise schedule (cosine, linear, sigmoid, custom)."""
    
    block_type = "noise/schedule/abstract"
    
    @classmethod
    def declare_io(cls) -> dict:
        return {
            "num_steps": InputPort("num_steps", data_type="scalar", description="Number of timesteps"),
            "timesteps": OutputPort("timesteps", data_type="tensor", description="Timestep schedule"),
        }
    
    def process(self, **port_inputs) -> dict:
        num_steps = port_inputs.get("num_steps", 50)
        timesteps = self.get_timesteps(int(num_steps))
        return {"timesteps": timesteps, "output": timesteps}
    
    def _forward_impl(self, *args, **kwargs):
        return None
    
    @abstractmethod
    def get_timesteps(self, num_steps: int) -> torch.Tensor:
        """Return a tensor of timesteps for the given number of steps."""
        pass
    
    @abstractmethod
    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """alpha(t) cumulative product for DDPM-style schedules."""
        pass
    
    @abstractmethod
    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """sigma(t) for variance."""
        pass


@register_block("noise/schedule/linear")
class LinearSchedule(NoiseSchedule):
    """Linear beta schedule (SD 1.5 / SDXL default).
    
    beta linearly increases from beta_start to beta_end over num_train_timesteps.
    alpha_cumprod = cumulative product of (1 - beta).
    """
    
    block_type = "noise/schedule/linear"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {})
        self.num_train_timesteps = int(self.config.get("num_train_timesteps", 1000))
        beta_start = float(self.config.get("beta_start", 0.00085))
        beta_end = float(self.config.get("beta_end", 0.012))
        
        # Scaled linear schedule (as in SD 1.5)
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, self.num_train_timesteps) ** 2
        alphas = 1.0 - betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
    
    def get_timesteps(self, num_steps: int) -> torch.Tensor:
        """Evenly spaced discrete timesteps from high noise to low noise."""
        step_ratio = self.num_train_timesteps // num_steps
        timesteps = torch.arange(num_steps) * step_ratio
        return timesteps.flip(0)
    
    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        t = t.long().clamp(0, self.num_train_timesteps - 1)
        return self.alphas_cumprod[t]
    
    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.get_alpha(t)).sqrt()


@register_block("noise/schedule/cosine")
class CosineSchedule(NoiseSchedule):
    """Cosine noise schedule (improved DDPM by Nichol & Dhariwal).
    
    alpha_cumprod(t) = cos^2(pi/2 * (t/T + s) / (1 + s))
    """
    
    block_type = "noise/schedule/cosine"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {})
        self.num_train_timesteps = int(self.config.get("num_train_timesteps", 1000))
        self.s = float(self.config.get("s", 0.008))  # offset to prevent singularity at t=0
        
        steps = torch.arange(self.num_train_timesteps + 1, dtype=torch.float64)
        f_t = torch.cos(((steps / self.num_train_timesteps) + self.s) / (1 + self.s) * math.pi * 0.5) ** 2
        alphas_cumprod = f_t / f_t[0]
        # Clip betas to prevent singularities
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(max=0.999)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
    
    def get_timesteps(self, num_steps: int) -> torch.Tensor:
        step_ratio = self.num_train_timesteps // num_steps
        timesteps = torch.arange(num_steps) * step_ratio
        return timesteps.flip(0)
    
    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        t = t.long().clamp(0, self.num_train_timesteps - 1)
        return self.alphas_cumprod[t]
    
    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.get_alpha(t)).sqrt()


@register_block("noise/schedule/sigmoid")
class SigmoidSchedule(NoiseSchedule):
    """Sigmoid noise schedule (used in SD3, Flux-style models).
    
    Operates in continuous time [0, 1]. Good for flow-matching.
    sigma(t) = sigmoid(start + t * (end - start))
    """
    
    block_type = "noise/schedule/sigmoid"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {})
        self.start = float(self.config.get("start", -3.0))
        self.end = float(self.config.get("end", 3.0))
        self.tau = float(self.config.get("tau", 1.0))  # temperature
    
    def get_timesteps(self, num_steps: int) -> torch.Tensor:
        """Continuous timesteps from 1.0 (noisy) to near 0.0 (clean)."""
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1)[:-1]
        return timesteps
    
    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """For flow-matching, alpha(t) = 1 - sigma(t)."""
        return 1.0 - self.get_sigma(t)
    
    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """sigma(t) via sigmoid mapping."""
        log_snr = self.start + t * (self.end - self.start)
        return torch.sigmoid(log_snr / self.tau)