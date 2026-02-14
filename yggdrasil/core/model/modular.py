# yggdrasil/core/model/modular.py
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Any, Dict, Optional, Tuple, List

from yggdrasil.core.block.base import AbstractBlock
from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.slot import Slot
from yggdrasil.core.block.port import Port, InputPort, OutputPort, TensorSpec


@register_block("model/modular")
class ModularDiffusionModel(AbstractBlock, nn.Module):
    """Модульная диффузионная модель — Lego-конструктор.
    
    Поддерживает два режима выполнения:
    1. Legacy (slot-based): Жёсткий pipeline encode → position → condition → backbone → guidance
    2. Graph-based: Выполнение через ComputeGraph с произвольным порядком
    
    Работает с любой модальностью и любыми адаптерами.
    """
    
    block_type = "model/modular"
    block_version = "2.0.0"
    
    def __init__(self, config: DictConfig | dict):
        AbstractBlock.__init__(self, config)
        self._cached_timestep_emb = None
        self.is_training = False
        self._compute_graph = None  # Optional graph for graph-based execution
    
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
        """Port-based execution for ComputeGraph."""
        x = port_inputs.get("x")
        t = port_inputs.get("t")
        condition = port_inputs.get("condition")
        result = self._forward_impl(x, t, condition, return_dict=True)
        result["output"] = result.get("noise_pred")
        return result
    
    def _define_slots(self) -> Dict[str, Slot]:
        """Определяем все Lego-слоты модели."""
        from .guidance import AbstractGuidance
        from .backbone import AbstractBackbone
        from .codec import AbstractLatentCodec
        from .conditioner import AbstractConditioner
        from .position import AbstractPositionEmbedder
        from yggdrasil.core.diffusion.process import AbstractDiffusionProcess
        
        return {
            "backbone": Slot(name="backbone", accepts=AbstractBackbone, multiple=False, optional=False),
            "codec": Slot(name="codec", accepts=AbstractLatentCodec, multiple=False, optional=True, default={"type": "codec/identity"}),
            "conditioner": Slot(name="conditioner", accepts=AbstractConditioner, multiple=True, optional=True),
            "guidance": Slot(name="guidance", accepts=AbstractGuidance, multiple=True, optional=True, default={"type": "guidance/cfg"}),
            "position": Slot(name="position", accepts=AbstractPositionEmbedder, multiple=False, optional=True, default={"type": "position/rope_nd"}),
            "adapters": Slot(name="adapters", accepts=AbstractBlock, multiple=True, optional=True),
            "diffusion_process": Slot(name="diffusion_process", accepts=AbstractDiffusionProcess, multiple=False, optional=True),
        }
    
    # ==================== GRAPH-BASED EXECUTION ====================
    
    def as_graph(self) -> "ComputeGraph":
        """Конвертировать slot-based модель в ComputeGraph.
        
        Позволяет перейти от legacy pipeline к graph-based execution.
        Возвращает ComputeGraph с узлами для каждого компонента.
        """
        from yggdrasil.core.graph.graph import ComputeGraph
        
        graph = ComputeGraph(f"model_{self.block_id}")
        
        # Add nodes for each component
        if "codec" in self._slot_children:
            graph.add_node("codec", self._slot_children["codec"])
        
        if self._slot_children.get("position") is not None:
            graph.add_node("position", self._slot_children["position"])
        
        for i, cond in enumerate(self._slot_children.get("conditioner", [])):
            graph.add_node(f"conditioner_{i}", cond)
        
        graph.add_node("backbone", self._slot_children["backbone"])
        
        for i, g in enumerate(self._slot_children.get("guidance", [])):
            # Set backbone reference for graph-mode CFG/SAG
            if hasattr(g, '_backbone_ref'):
                g._backbone_ref = self._slot_children["backbone"]
            graph.add_node(f"guidance_{i}", g)
        
        for i, a in enumerate(self._slot_children.get("adapters", [])):
            graph.add_node(f"adapter_{i}", a)
        
        # Wire connections
        graph.expose_input("x", "backbone", "x")
        graph.expose_input("timestep", "backbone", "timestep")
        
        if "position" in graph.nodes:
            graph.expose_input("timestep_pos", "position", "timestep")
            graph.connect("position", "embedding", "backbone", "position_embedding")
        
        for i in range(len(self._slot_children.get("conditioner", []))):
            graph.expose_input(f"condition_{i}", f"conditioner_{i}", "raw_condition")
            graph.connect(f"conditioner_{i}", "embedding", "backbone", "condition")
        
        # Guidance chain — also wire x, timestep, condition for CFG dual-pass
        prev_node = "backbone"
        prev_port = "output"
        for i in range(len(self._slot_children.get("guidance", []))):
            g_node = f"guidance_{i}"
            graph.connect(prev_node, prev_port, g_node, "model_output")
            # Fan-out x and timestep to guidance for internal dual-pass
            graph.expose_input("x", g_node, "x")
            graph.expose_input("timestep", g_node, "t")
            prev_node = g_node
            prev_port = "guided_output"
        
        graph.expose_output("noise_pred", prev_node, prev_port)
        
        self._compute_graph = graph
        return graph
    
    # ==================== LEGACY FORWARD ====================
    
    def _forward_impl(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Dict[str, Any] | None = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        """Основной forward — работает с ЛЮБОЙ модальностью."""
        backbone_param = next(self._slot_children["backbone"].parameters(), None)
        if backbone_param is not None:
            device, dtype = backbone_param.device, backbone_param.dtype
            x = x.to(device=device, dtype=dtype)
            t = t.to(device=device)
        else:
            x = x.to(dtype=torch.float32)

        # 1. Encode to latent if needed
        if self.has_slot("codec"):
            codec = self._slot_children["codec"]
            latent_ch = getattr(codec, "latent_channels", 4)
            if x.shape[1] == latent_ch:
                latents = x
            else:
                latents = self._encode(x)
        else:
            latents = x
        
        # 2. Position embeddings
        pos_emb = None
        if self._slot_children.get("position") is not None:
            pos_emb = self._slot_children["position"](t, latents.shape)
        
        # 3. Process conditions
        cond_emb = self._process_conditions(condition) if condition else None
        
        # 4. Backbone forward
        backbone_output = self._slot_children["backbone"](
            latents, timestep=t, condition=cond_emb, position_embedding=pos_emb
        )
        
        # 5. Apply guidance
        model_output = self._apply_guidance(backbone_output, condition, latents, t)
        
        if return_dict:
            return {
                "noise_pred": model_output,
                "x0_pred": self._predict_x0(model_output, latents, t),
                "velocity": self._predict_velocity(model_output, latents, t),
                "latents": latents
            }
        return model_output
    
    # ==================== GUIDANCE ====================
    
    def _apply_guidance(self, output, condition, x, t):
        result = output
        for guidance in self._slot_children.get("guidance", []):
            result = guidance(result, condition=condition, model=self, x=x, t=t)
        return result
    
    # ==================== HELPERS ====================
    
    def _encode(self, x):
        if "codec" in self._slot_children:
            return self._slot_children["codec"].encode(x)
        return x
    
    def _decode_output(self, output):
        if "codec" in self._slot_children:
            return self._slot_children["codec"].decode(output)
        return output
    
    def _process_conditions(self, condition):
        cond_emb = {}
        for conditioner in self._slot_children.get("conditioner", []):
            emb = conditioner(condition)
            cond_emb.update(emb)
        return cond_emb
    
    def _predict_x0(self, noise_pred, x, t):
        if "diffusion_process" in self._slot_children:
            return self._slot_children["diffusion_process"].predict_x0(noise_pred, x, t)
        alpha = torch.cos(t * 0.5 * torch.pi) ** 2
        return (x - (1 - alpha).sqrt() * noise_pred) / alpha.sqrt()
    
    def _predict_velocity(self, noise_pred, x, t):
        if "diffusion_process" in self._slot_children:
            return self._slot_children["diffusion_process"].predict_velocity(noise_pred, x, t)
        return noise_pred
    
    # ==================== LEGO API ====================
    
    def has_slot(self, slot_name):
        return slot_name in self.slots
    
    def attach_adapter(self, adapter):
        self.attach_slot("adapters", adapter)
        if hasattr(adapter, "inject_into"):
            adapter.inject_into(self._slot_children["backbone"])
    
    def set_training_mode(self, mode=True):
        self.is_training = mode
        self.train(mode)
    
    def encode(self, data):
        return self._encode(data)
    
    def decode(self, latents):
        codec = self._slot_children.get("codec")
        if codec is not None:
            return codec.decode(latents)
        return latents
    
    def forward_for_loss(self, x, t, condition=None):
        self.set_training_mode(True)
        return self._forward_impl(x, t, condition, return_dict=True)
    
    def generate(self, condition, **kwargs):
        from yggdrasil.core.engine.sampler import DiffusionSampler
        sampler = DiffusionSampler({"model": self, **kwargs})
        return sampler.sample(condition=condition, **kwargs)
    
    def __repr__(self):
        slots = [f"{k}={len(v) if isinstance(v, list) else 1}" for k, v in self._slot_children.items()]
        return f"<ModularDiffusionModel {self.block_id} | {' | '.join(slots)}>"
