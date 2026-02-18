import torch
import torch.nn.functional as F
from abc import abstractmethod
from typing import Optional, Tuple
from omegaconf import DictConfig

from ....core.block.base import AbstractBaseBlock
from ....core.block.registry import register_block
from ....core.block.port import Port, InputPort, OutputPort, TensorSpec


@register_block("noise/sampler/abstract")
class NoiseSampler(AbstractBaseBlock):
    """Abstract noise generator (Gaussian, Pyramid, Offset, etc.)."""
    
    block_type = "noise/sampler/abstract"
    
    @classmethod
    def declare_io(cls) -> dict:
        return {
            "shape": InputPort("shape", data_type="any", description="Output tensor shape"),
            "device": InputPort("device", data_type="any", optional=True, description="Target device"),
            "generator": InputPort("generator", data_type="any", optional=True, description="RNG generator"),
            "noise": OutputPort("noise", spec=TensorSpec(space="noise"), description="Generated noise"),
        }
    
    def process(self, **port_inputs) -> dict:
        import torch
        shape = port_inputs.get("shape", (1, 4, 64, 64))
        device = port_inputs.get("device", torch.device("cpu"))
        generator = port_inputs.get("generator")
        noise = self.sample(shape, device, generator)
        return {"noise": noise, "output": noise}
    
    def _forward_impl(self, *args, **kwargs):
        return self.sample(*args, **kwargs)
    
    @abstractmethod
    def sample(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Return a noise tensor of the given shape."""
        pass


@register_block("noise/sampler/gaussian")
class GaussianNoiseSampler(NoiseSampler):
    """Standard Gaussian noise N(0, 1)."""
    
    block_type = "noise/sampler/gaussian"
    
    def sample(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        return torch.randn(shape, device=device, generator=generator)


@register_block("noise/sampler/pyramid")
class PyramidNoiseSampler(NoiseSampler):
    """Multi-scale pyramid noise for better structural coherence.
    
    Generates noise at multiple resolutions and combines them.
    Useful for high-resolution generation.
    """
    
    block_type = "noise/sampler/pyramid"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {})
        self.discount = float(self.config.get("discount", 0.8))
        self.num_levels = int(self.config.get("num_levels", 4))
    
    def sample(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        # Start with base noise
        noise = torch.randn(shape, device=device, generator=generator)
        
        b, c = shape[0], shape[1]
        spatial = shape[2:]  # works for any dimensionality
        
        for level in range(1, self.num_levels):
            scale = 2 ** level
            # Compute downscaled spatial dims
            small_spatial = tuple(max(1, s // scale) for s in spatial)
            small_shape = (b, c) + small_spatial
            
            # Generate low-res noise and upscale
            low_noise = torch.randn(small_shape, device=device, generator=generator)
            
            # Upscale to original size (works for 1D, 2D, 3D)
            if len(spatial) == 1:
                upscaled = F.interpolate(low_noise, size=spatial, mode="linear", align_corners=False)
            elif len(spatial) == 2:
                upscaled = F.interpolate(low_noise, size=spatial, mode="bilinear", align_corners=False)
            elif len(spatial) == 3:
                upscaled = F.interpolate(low_noise, size=spatial, mode="trilinear", align_corners=False)
            else:
                upscaled = low_noise  # fallback for exotic dims
            
            noise = noise + upscaled * (self.discount ** level)
        
        # Normalize to unit variance
        noise = noise / noise.std()
        return noise


@register_block("noise/sampler/offset")
class OffsetNoiseSampler(NoiseSampler):
    """Offset noise for better dark/light image generation.
    
    Adds a small channel-wise offset to standard Gaussian noise,
    as described in "Common Diffusion Noise Schedules and Sample Steps are Flawed".
    """
    
    block_type = "noise/sampler/offset"
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {})
        self.offset_weight = float(self.config.get("offset_weight", 0.1))
    
    def sample(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        noise = torch.randn(shape, device=device, generator=generator)
        
        # Channel-wise offset: same offset for all spatial positions
        # shape: (B, C, *spatial) -> offset shape: (B, C, 1, 1, ...)
        offset_shape = (shape[0], shape[1]) + (1,) * (len(shape) - 2)
        offset = torch.randn(offset_shape, device=device, generator=generator)
        
        noise = noise + self.offset_weight * offset
        return noise