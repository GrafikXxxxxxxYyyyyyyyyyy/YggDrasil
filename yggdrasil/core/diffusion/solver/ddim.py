import math
import torch
from omegaconf import DictConfig

from .base import AbstractSolver
from ....core.block.registry import register_block


@register_block("diffusion/solver/ddim")
class DDIMSolver(AbstractSolver):
    """DDIM solver with built-in noise schedule.
    
    Supports linear, scaled_linear (SD/LDM), and cosine. SD 1.5 uses
    scaled_linear with beta_start=0.00085, beta_end=0.012.
    
    Config:
        eta: float (default 0.0) — stochasticity. 0 = deterministic DDIM.
        beta_schedule: str ("linear" | "scaled_linear" | "cosine") — default "scaled_linear"
        beta_start: float (default 0.00085)
        beta_end: float (default 0.012)
        num_train_timesteps: int (default 1000)
    
    Formula (eta=0, deterministic):
        x_{t-1} = sqrt(alpha_{t-1}) * pred_x0
                + sqrt(1 - alpha_{t-1}) * predicted_noise
    """
    
    block_type = "diffusion/solver/ddim"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.eta = config.get("eta", 0.0)
        self.num_train_timesteps = int(config.get("num_train_timesteps", 1000))
        # Default 2.0: SD uses 1.0 in diffusers but aggressive clamp can distort content in our pipeline
        self.clip_sample_range = float(config.get("clip_sample_range", 2.0))
        
        # Build noise schedule (alphas_cumprod) — formulas match HuggingFace diffusers
        beta_schedule = config.get("beta_schedule", "scaled_linear")
        beta_start = float(config.get("beta_start", 0.00085))
        beta_end = float(config.get("beta_end", 0.012))
        
        if beta_schedule == "linear":
            # Linear in beta (diffusers "linear")
            betas = torch.linspace(beta_start, beta_end, self.num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            # Linear in sqrt(beta) then square — used by Stable Diffusion / LDM
            betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, self.num_train_timesteps
            ) ** 2
        elif beta_schedule == "cosine":
            steps = self.num_train_timesteps + 1
            t = torch.linspace(0, self.num_train_timesteps, steps) / self.num_train_timesteps
            alphas_bar = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            betas = betas.clamp(max=0.999)
        elif beta_schedule == "squaredcos_cap_v2":
            # Improved cosine schedule (used by some models)
            steps = self.num_train_timesteps + 1
            t = torch.linspace(0, self.num_train_timesteps, steps) / self.num_train_timesteps
            alphas_bar = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            betas = betas.clamp(max=0.999)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Register as buffer so .to(device) moves it automatically
        self.register_buffer("alphas_cumprod", alphas_cumprod)
    
    def _get_alpha(self, timestep: torch.Tensor) -> torch.Tensor:
        """Get alpha_cumprod for a discrete timestep."""
        t = timestep.long().clamp(0, self.num_train_timesteps - 1)
        return self.alphas_cumprod[t]
    
    def step(self, model_output, current_latents, timestep, process=None, **kwargs):
        next_timestep = kwargs.get("next_timestep")

        # Run entire step in float32; use schedule on same device as latents (avoids cross-device index bugs)
        device = current_latents.device
        current_latents = current_latents.to(device=device, dtype=torch.float32)
        model_output = model_output.to(device=device, dtype=torch.float32)
        alphas_cumprod = self.alphas_cumprod.to(device=device, dtype=torch.float32)

        # Get alpha cumprod — from external process or built-in schedule
        if process is not None and hasattr(process, 'get_alpha'):
            alpha_prod_t = process.get_alpha(timestep)
            alpha_prod_t_prev = (
                process.get_alpha(next_timestep) if next_timestep is not None
                else torch.ones(1)
            )
        else:
            # Use built-in schedule (index on same device as latents)
            t_idx = timestep.long().clamp(0, self.num_train_timesteps - 1).to(device)
            alpha_prod_t = alphas_cumprod[t_idx]
            if next_timestep is None:
                alpha_prod_t_prev = torch.ones(1, device=device, dtype=torch.float32)
            else:
                is_final = (
                    (isinstance(next_timestep, int) and next_timestep == 0)
                    or (isinstance(next_timestep, torch.Tensor) and next_timestep.numel() == 1 and next_timestep.item() == 0)
                )
                if is_final:
                    alpha_prod_t_prev = torch.ones(1, device=device, dtype=torch.float32)
                else:
                    next_idx = next_timestep.long().clamp(0, self.num_train_timesteps - 1).to(device)
                    alpha_prod_t_prev = alphas_cumprod[next_idx]
        
        alpha_prod_t = alpha_prod_t.to(device=device, dtype=torch.float32)
        alpha_prod_t_prev = alpha_prod_t_prev.to(device=device, dtype=torch.float32)

        # Reshape for broadcasting
        while alpha_prod_t.dim() < current_latents.dim():
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
            alpha_prod_t_prev = alpha_prod_t_prev.unsqueeze(-1)

        # Predict x0 from epsilon (same as diffusers prediction_type="epsilon")
        pred_x0 = (current_latents - (1 - alpha_prod_t).sqrt() * model_output) / alpha_prod_t.sqrt().clamp(min=1e-8)
        pred_x0 = pred_x0.clamp(-self.clip_sample_range, self.clip_sample_range)

        # DDIM step (eta=0 → deterministic)
        pred_direction = (1 - alpha_prod_t_prev).sqrt() * model_output
        next_latents = alpha_prod_t_prev.sqrt() * pred_x0 + pred_direction

        # Stochastic part (eta > 0)
        if self.eta > 0:
            variance = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t).clamp(min=1e-8)) * \
                       (1 - alpha_prod_t / alpha_prod_t_prev.clamp(min=1e-8))
            sigma = self.eta * variance.clamp(min=0).sqrt()
            noise = torch.randn_like(current_latents, device=device, dtype=torch.float32)
            next_latents = next_latents + sigma * noise

        return next_latents
