"""MetricBlock — evaluation metrics as graph nodes.

Metrics can be part of the training or evaluation graph.
They accumulate values across batches and compute final metrics.
"""
from __future__ import annotations

import math
import torch
from typing import Any, Dict, Optional
from omegaconf import DictConfig

from ...core.block.base import AbstractBaseBlock
from ...core.block.registry import register_block
from ...core.block.port import InputPort, OutputPort, Port


@register_block("metric/psnr")
class PSNRBlock(AbstractBaseBlock):
    """Peak Signal-to-Noise Ratio metric."""
    
    block_type = "metric/psnr"
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "metric/psnr"}
        super().__init__(config)
        self.max_val = float(self.config.get("max_val", 1.0))
        self._accumulated = []
    
    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        return {
            "prediction": InputPort("prediction"),
            "target": InputPort("target"),
            "value": OutputPort("value", data_type="tensor"),
        }
    
    def process(self, **kw) -> dict:
        pred = kw["prediction"]
        target = kw["target"].to(pred.device)
        mse = torch.mean((pred - target) ** 2)
        psnr = 10.0 * torch.log10(self.max_val ** 2 / (mse + 1e-10))
        self._accumulated.append(psnr.item())
        return {"value": psnr, "output": psnr}
    
    def reset(self):
        self._accumulated = []
    
    def compute(self) -> float:
        if not self._accumulated:
            return 0.0
        return sum(self._accumulated) / len(self._accumulated)
    
    def _forward_impl(self, **kwargs):
        return self.process(**kwargs)


@register_block("metric/ssim")
class SSIMBlock(AbstractBaseBlock):
    """Structural Similarity Index (stub — full impl needs gaussian kernel)."""
    
    block_type = "metric/ssim"
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "metric/ssim"}
        super().__init__(config)
        self._accumulated = []
    
    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        return {
            "prediction": InputPort("prediction"),
            "target": InputPort("target"),
            "value": OutputPort("value", data_type="tensor"),
        }
    
    def process(self, **kw) -> dict:
        pred = kw["prediction"]
        target = kw["target"].to(pred.device)
        # Simplified SSIM (channel-wise mean)
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu_x = pred.mean(dim=(-2, -1), keepdim=True)
        mu_y = target.mean(dim=(-2, -1), keepdim=True)
        sigma_x = ((pred - mu_x) ** 2).mean(dim=(-2, -1), keepdim=True)
        sigma_y = ((target - mu_y) ** 2).mean(dim=(-2, -1), keepdim=True)
        sigma_xy = ((pred - mu_x) * (target - mu_y)).mean(dim=(-2, -1), keepdim=True)
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
        val = ssim.mean()
        self._accumulated.append(val.item())
        return {"value": val, "output": val}
    
    def reset(self):
        self._accumulated = []
    
    def compute(self) -> float:
        if not self._accumulated:
            return 0.0
        return sum(self._accumulated) / len(self._accumulated)
    
    def _forward_impl(self, **kwargs):
        return self.process(**kwargs)


@register_block("metric/fid_accumulator")
class FIDAccumulatorBlock(AbstractBaseBlock):
    """FID metric accumulator — collects features for computing FID.
    
    FID is computed offline after accumulating enough features.
    This block accumulates inception features.
    """
    
    block_type = "metric/fid_accumulator"
    
    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "metric/fid_accumulator"}
        super().__init__(config)
        self._real_features = []
        self._fake_features = []
    
    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        return {
            "real_features": InputPort("real_features"),
            "fake_features": InputPort("fake_features"),
            "value": OutputPort("value", data_type="tensor"),
        }
    
    def process(self, **kw) -> dict:
        real = kw.get("real_features")
        fake = kw.get("fake_features")
        if real is not None:
            self._real_features.append(real.detach().cpu())
        if fake is not None:
            self._fake_features.append(fake.detach().cpu())
        # Return placeholder (actual FID computed offline)
        return {"value": torch.tensor(0.0), "output": torch.tensor(0.0)}
    
    def compute(self) -> float:
        """Compute FID from accumulated features."""
        if not self._real_features or not self._fake_features:
            return float('inf')
        
        real = torch.cat(self._real_features, dim=0).numpy()
        fake = torch.cat(self._fake_features, dim=0).numpy()
        
        import numpy as np
        mu_r, sigma_r = real.mean(axis=0), np.cov(real, rowvar=False)
        mu_f, sigma_f = fake.mean(axis=0), np.cov(fake, rowvar=False)
        
        diff = mu_r - mu_f
        from scipy import linalg
        covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean)
        return float(fid)
    
    def reset(self):
        self._real_features = []
        self._fake_features = []
    
    def _forward_impl(self, **kwargs):
        return self.process(**kwargs)
