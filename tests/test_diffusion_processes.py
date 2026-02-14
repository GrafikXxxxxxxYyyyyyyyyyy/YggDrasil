"""Tests for diffusion processes: DDPM, RectifiedFlow, Consistency."""
import pytest
import torch

from yggdrasil.core.block.registry import auto_discover


@pytest.fixture(autouse=True)
def discover():
    auto_discover()


class TestDDPM:
    def test_forward_process(self):
        from yggdrasil.core.diffusion.ddpm import DDPMProcess
        
        process = DDPMProcess({"type": "diffusion/process/ddpm"})
        x0 = torch.randn(2, 4, 8, 8)
        t = torch.tensor([500, 200])
        
        result = process.forward_process(x0, t)
        assert "xt" in result
        assert "target" in result
        assert result["xt"].shape == x0.shape
    
    def test_predict_x0(self):
        from yggdrasil.core.diffusion.ddpm import DDPMProcess
        
        process = DDPMProcess({"type": "diffusion/process/ddpm"})
        model_output = torch.randn(2, 4, 8, 8)
        xt = torch.randn(2, 4, 8, 8)
        t = torch.tensor([500, 200])
        
        x0 = process.predict_x0(model_output, xt, t)
        assert x0.shape == xt.shape


class TestRectifiedFlow:
    def test_forward_process(self):
        from yggdrasil.core.diffusion.flow import RectifiedFlowProcess
        
        process = RectifiedFlowProcess({"type": "diffusion/process/flow/rectified"})
        x0 = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.5, 0.3]).reshape(-1, 1, 1, 1)
        
        result = process.forward_process(x0, t)
        assert "xt" in result
        assert result["xt"].shape == x0.shape
    
    def test_predict_velocity(self):
        from yggdrasil.core.diffusion.flow import RectifiedFlowProcess
        
        process = RectifiedFlowProcess({"type": "diffusion/process/flow/rectified"})
        model_output = torch.randn(2, 4, 8, 8)
        xt = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.5, 0.3]).reshape(-1, 1, 1, 1)
        
        v = process.predict_velocity(model_output, xt, t)
        assert v.shape == model_output.shape


class TestConsistency:
    def test_forward_process(self):
        from yggdrasil.core.diffusion.consistency import ConsistencyProcess
        
        process = ConsistencyProcess({"type": "diffusion/process/consistency"})
        x0 = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.5, 0.3])
        
        result = process.forward_process(x0, t)
        assert "xt" in result
    
    def test_reverse_step(self):
        from yggdrasil.core.diffusion.consistency import ConsistencyProcess
        
        process = ConsistencyProcess({"type": "diffusion/process/consistency"})
        model_output = torch.randn(2, 4, 8, 8)
        xt = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.5])
        
        result = process.reverse_step(model_output, xt, t)
        # Consistency models predict x0 directly
        assert result.shape == xt.shape


class TestNoiseSchedules:
    def test_linear_schedule(self):
        from yggdrasil.core.diffusion.noise.schedule import LinearSchedule
        
        sched = LinearSchedule({"num_train_timesteps": 1000})
        timesteps = sched.get_timesteps(50)
        assert timesteps.shape[0] == 50
        assert timesteps[0] > timesteps[-1]  # descending
    
    def test_cosine_schedule(self):
        from yggdrasil.core.diffusion.noise.schedule import CosineSchedule
        
        sched = CosineSchedule({"num_train_timesteps": 1000})
        timesteps = sched.get_timesteps(50)
        assert timesteps.shape[0] == 50
        
        alpha = sched.get_alpha(torch.tensor([0]))
        assert alpha.item() > 0.9  # alpha should be close to 1 at t=0
    
    def test_sigmoid_schedule(self):
        from yggdrasil.core.diffusion.noise.schedule import SigmoidSchedule
        
        sched = SigmoidSchedule({})
        timesteps = sched.get_timesteps(50)
        assert timesteps.shape[0] == 50
        assert timesteps[0] > timesteps[-1]


class TestNoiseSamplers:
    def test_gaussian(self):
        from yggdrasil.core.diffusion.noise.sampler import GaussianNoiseSampler
        
        sampler = GaussianNoiseSampler({})
        noise = sampler.sample((2, 4, 8, 8), device=torch.device("cpu"))
        assert noise.shape == (2, 4, 8, 8)
    
    def test_pyramid(self):
        from yggdrasil.core.diffusion.noise.sampler import PyramidNoiseSampler
        
        sampler = PyramidNoiseSampler({"num_levels": 3})
        noise = sampler.sample((2, 4, 16, 16), device=torch.device("cpu"))
        assert noise.shape == (2, 4, 16, 16)
    
    def test_offset(self):
        from yggdrasil.core.diffusion.noise.sampler import OffsetNoiseSampler
        
        sampler = OffsetNoiseSampler({"offset_weight": 0.1})
        noise = sampler.sample((2, 4, 8, 8), device=torch.device("cpu"))
        assert noise.shape == (2, 4, 8, 8)
