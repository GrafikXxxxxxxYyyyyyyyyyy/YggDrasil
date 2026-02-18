# yggdrasil/core/model/modular.py
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Any, Dict, Optional

from yggdrasil.core.block.base import AbstractBaseBlock
from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import Port, InputPort, OutputPort, TensorSpec
from yggdrasil.core.graph.graph import ComputeGraph
from yggdrasil.core.graph.model_graph_builder import build_model_graph
from yggdrasil.core.graph.executor import GraphExecutor


@register_block("model/modular")
class ModularDiffusionModel(AbstractBaseBlock, nn.Module):
    """Modular diffusion model — graph-only (no slots).

    The model is a ComputeGraph of blocks (codec, position, conditioners,
    backbone, guidance, adapters). Built from config via build_model_graph();
    execution is always through GraphExecutor.
    """

    block_type = "model/modular"
    block_version = "2.0.0"

    def __init__(self, config: DictConfig | dict):
        nn.Module.__init__(self)
        super(AbstractBaseBlock, self).__init__()
        from omegaconf import OmegaConf
        self.config = OmegaConf.create(config) if isinstance(config, dict) else config
        self.block_id = self.config.get("id", f"{self.block_type}_{id(self)}")
        self.pre_hooks: list = []
        self.post_hooks: list = []
        self._cached_timestep_emb = None
        self.is_training = False

        # Build model as a graph (no slots)
        self._graph: ComputeGraph = build_model_graph(self.config, name=f"model_{self.block_id}")
        self._executor = GraphExecutor(no_grad=False)  # model may be trained

        # Register graph nodes as submodules for .to(device) and state_dict
        for name, block in self._graph.nodes.items():
            if isinstance(block, nn.Module):
                self.add_module(f"_graph_{name}", block)

    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        return {
            "x": InputPort("x", spec=TensorSpec(space="latent"), description="Input latents or pixels"),
            "t": InputPort("t", data_type="tensor", description="Timestep"),
            "condition": InputPort("condition", data_type="dict", optional=True, description="Condition dict"),
            "noise_pred": OutputPort("noise_pred", spec=TensorSpec(space="latent"), description="Predicted noise"),
            "x0_pred": OutputPort("x0_pred", spec=TensorSpec(space="latent"), description="Predicted clean data"),
            "velocity": OutputPort("velocity", description="Predicted velocity"),
            "latents": OutputPort("latents", spec=TensorSpec(space="latent"), description="Current latents"),
        }

    def process(self, **port_inputs) -> Dict[str, Any]:
        """Port-based execution via graph."""
        x = port_inputs.get("x")
        t = port_inputs.get("t")
        condition = port_inputs.get("condition")
        result = self._forward_impl(x, t, condition, return_dict=True)
        result["output"] = result.get("noise_pred")
        return result

    def _forward_impl(self, x: torch.Tensor, t: torch.Tensor, condition: Dict[str, Any] | None = None, return_dict: bool = True, **kwargs) -> Dict[str, torch.Tensor] | torch.Tensor:
        """Forward: run the model graph."""
        nodes = self._graph.nodes
        backbone = nodes["backbone"]
        backbone_param = next(backbone.parameters(), None)
        if backbone_param is not None:
            device, dtype = backbone_param.device, backbone_param.dtype
            x = x.to(device=device, dtype=dtype)
            t = t.to(device=device)
        else:
            x = x.to(dtype=torch.float32)

        # Encode to latent if codec present
        if "codec" in nodes:
            codec = nodes["codec"]
            latent_ch = getattr(codec, "latent_channels", 4)
            if x.shape[1] == latent_ch:
                latents = x
            else:
                latents = self._encode(x)
        else:
            latents = x

        inputs = {"x": latents, "timestep": t, "condition": condition or {}}
        with torch.set_grad_enabled(self.is_training):
            out = self._executor.execute_training(self._graph, **inputs)
        noise_pred = out.get("noise_pred")
        if noise_pred is None:
            noise_pred = out.get("output")

        x0_pred = self._predict_x0(noise_pred, latents, t)
        velocity = self._predict_velocity(noise_pred, latents, t)

        if return_dict:
            return {
                "noise_pred": noise_pred,
                "x0_pred": x0_pred,
                "velocity": velocity,
                "latents": latents,
            }
        return noise_pred

    def _predict_x0(self, noise_pred: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if "diffusion_process" in self.config and self.config.diffusion_process:
            from yggdrasil.core.block.builder import BlockBuilder
            proc = BlockBuilder.build(self.config.diffusion_process)
            return proc.predict_x0(noise_pred, x, t)
        alpha = torch.cos(t * 0.5 * torch.pi) ** 2
        return (x - (1 - alpha).sqrt() * noise_pred) / alpha.sqrt()

    def _predict_velocity(self, noise_pred: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if "diffusion_process" in self.config and self.config.diffusion_process:
            from yggdrasil.core.block.builder import BlockBuilder
            proc = BlockBuilder.build(self.config.diffusion_process)
            return proc.predict_velocity(noise_pred, x, t)
        return noise_pred

    # ---------- Public API (no slots) ----------

    def as_graph(self) -> ComputeGraph:
        """Return the internal model graph (for pipeline integration)."""
        return self._graph

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        if "codec" in self._graph.nodes:
            return self._graph.nodes["codec"].encode(data)
        return data

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if "codec" in self._graph.nodes:
            return self._graph.nodes["codec"].decode(latents)
        return latents

    def attach_adapter(self, adapter: AbstractBaseBlock) -> None:
        """Add an adapter to the model graph and wire it to the backbone."""
        nodes = self._graph.nodes
        n = sum(1 for k in nodes if k.startswith("adapter_"))
        node_name = f"adapter_{n}"
        self._graph.add_node(node_name, adapter)
        self._graph.connect(node_name, "output", "backbone", "adapter_features")
        if isinstance(adapter, nn.Module):
            self.add_module(f"_graph_{node_name}", adapter)
        if hasattr(adapter, "inject_into"):
            adapter.inject_into(nodes["backbone"])

    def set_training_mode(self, mode: bool = True) -> None:
        self.is_training = mode
        self.train(mode)

    def forward_for_loss(self, x: torch.Tensor, t: torch.Tensor, condition: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        self.set_training_mode(True)
        out = self._forward_impl(x, t, condition, return_dict=True)
        return out if isinstance(out, dict) else {"noise_pred": out}

    def generate(self, condition: Dict[str, Any], **kwargs) -> Any:
        from yggdrasil.core.engine.sampler import DiffusionSampler
        sampler = DiffusionSampler({"model": self, **kwargs})
        return sampler.sample(condition=condition, **kwargs)

    def __repr__(self) -> str:
        node_summary = " | ".join(f"{k}=1" for k in self._graph.nodes)
        return f"<ModularDiffusionModel {self.block_id} | {node_summary}>"
