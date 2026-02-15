# yggdrasil/core/graph/subgraph.py
"""SubGraph — вложенный граф, работающий как единый блок.

SubGraph позволяет использовать целый граф как один узел
в другом графе. Это ключевой механизм для вложенности:
- DenoiseLoop = SubGraph с внутренним графом шагов
- CascadePipeline = SubGraph из нескольких stage
- Multi-modal pipeline = SubGraph из нескольких модальных графов
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from omegaconf import DictConfig, OmegaConf

from yggdrasil.core.block.base import AbstractBaseBlock
from yggdrasil.core.block.port import Port, InputPort, OutputPort
from yggdrasil.core.block.registry import register_block


@register_block("graph/subgraph")
class SubGraph(AbstractBaseBlock):
    """Вложенный граф, работающий как единый блок.
    
    Порты SubGraph = exposed inputs/outputs внутреннего графа.
    Выполнение SubGraph = выполнение внутреннего графа.
    
    Пример::
    
        inner = ComputeGraph("denoise_step")
        inner.add_node("backbone", unet_block)
        inner.add_node("guidance", cfg_block)
        inner.connect("backbone", "output", "guidance", "model_output")
        inner.expose_input("x", "backbone", "x")
        inner.expose_input("timestep", "backbone", "timestep")
        inner.expose_output("output", "guidance", "guided_output")
        
        # Use as a single block in another graph
        outer.add_node("denoise_step", SubGraph.from_graph(inner))
    """
    
    block_type = "graph/subgraph"
    block_version = "1.0.0"
    
    def __init__(self, config: DictConfig | dict):
        self._inner_graph = None  # Will be set by from_graph() or build
        super().__init__(config)
    
    @classmethod
    def from_graph(cls, graph: "ComputeGraph", name: str | None = None) -> SubGraph:
        """Создать SubGraph из ComputeGraph."""
        config = {"type": "graph/subgraph", "id": name or graph.name}
        instance = cls(config)
        instance._inner_graph = graph
        return instance
    
    @property
    def graph(self) -> "ComputeGraph":
        if self._inner_graph is None:
            from .graph import ComputeGraph
            self._inner_graph = ComputeGraph(self.block_id)
        return self._inner_graph
    
    @graph.setter
    def graph(self, value: "ComputeGraph"):
        self._inner_graph = value
    
    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        """Порты определяются динамически из внутреннего графа."""
        # Статически не можем определить — порты зависят от графа
        return {}
    
    def get_dynamic_ports(self) -> Dict[str, Port]:
        """Получить порты, основанные на exposed inputs/outputs графа."""
        ports = {}
        
        if self._inner_graph is not None:
            for input_name in self._inner_graph.graph_inputs:
                ports[input_name] = InputPort(input_name, data_type="any",
                                              description=f"Fan-out to {len(self._inner_graph.graph_inputs[input_name])} targets")
            for output_name in self._inner_graph.graph_outputs:
                ports[output_name] = OutputPort(output_name, data_type="any")
        
        return ports
    
    def get_input_ports(self) -> Dict[str, Port]:
        """Входные порты из exposed inputs графа."""
        ports = self.get_dynamic_ports()
        return {k: v for k, v in ports.items() if v.direction == "input"}
    
    def get_output_ports(self) -> Dict[str, Port]:
        """Выходные порты из exposed outputs графа."""
        ports = self.get_dynamic_ports()
        return {k: v for k, v in ports.items() if v.direction == "output"}
    
    def process(self, **port_inputs: Any) -> Dict[str, Any]:
        """Выполнить внутренний граф."""
        from .executor import GraphExecutor
        executor = GraphExecutor(no_grad=not self.training if hasattr(self, 'training') else True)
        return executor.execute(self.graph, **port_inputs)
    
    def _forward_impl(self, **kwargs) -> Any:
        """Fallback для совместимости с AbstractBaseBlock."""
        return self.process(**kwargs)
    
    def __repr__(self) -> str:
        n_nodes = len(self.graph.nodes) if self._inner_graph else 0
        return f"<SubGraph '{self.block_id}' nodes={n_nodes}>"


@register_block("graph/loop")
class LoopSubGraph(AbstractBaseBlock):
    """Цикл как блок: выполняет внутренний граф N раз.
    
    Ключевой блок для sampling loop. На каждой итерации:
    1. Подготавливает входы (латенты + текущий таймстеп)
    2. Выполняет внутренний граф (backbone -> guidance -> solver)
    3. Обновляет латенты из выхода solver
    
    Пример (denoise loop)::
    
        loop = LoopSubGraph.create(
            inner_graph=denoise_step_graph,
            num_iterations=50,
            iterate_over="timesteps",
            carry_vars=["latents"],
        )
    """
    
    block_type = "graph/loop"
    block_version = "1.0.0"
    
    def __init__(self, config: DictConfig | dict):
        self._inner_graph = None
        super().__init__(config)
        self.num_iterations = self.config.get("num_iterations", 50)
        self.carry_vars = list(self.config.get("carry_vars", ["latents"]))
        self.show_progress = self.config.get("show_progress", True)
        self.num_train_timesteps = int(self.config.get("num_train_timesteps", 1000))
        self.timestep_spacing = self.config.get("timestep_spacing", "leading")
        self.steps_offset = int(self.config.get("steps_offset", 0))
    
    @classmethod
    def create(
        cls,
        inner_graph: "ComputeGraph",
        num_iterations: int = 50,
        carry_vars: list | None = None,
        show_progress: bool = True,
        num_train_timesteps: int = 1000,
        timestep_spacing: str = "leading",
        steps_offset: int = 0,
    ) -> LoopSubGraph:
        """Создать LoopSubGraph."""
        config = {
            "type": "graph/loop",
            "num_iterations": num_iterations,
            "carry_vars": carry_vars or ["latents"],
            "show_progress": show_progress,
            "num_train_timesteps": num_train_timesteps,
            "timestep_spacing": timestep_spacing,
            "steps_offset": steps_offset,
        }
        instance = cls(config)
        instance._inner_graph = inner_graph
        return instance
    
    @property
    def graph(self) -> "ComputeGraph":
        if self._inner_graph is None:
            from .graph import ComputeGraph
            self._inner_graph = ComputeGraph(f"{self.block_id}_inner")
        return self._inner_graph
    
    @graph.setter
    def graph(self, value: "ComputeGraph"):
        self._inner_graph = value
    
    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        """Базовые порты цикла."""
        return {
            "initial_latents": InputPort("initial_latents", description="Начальные латенты (шум)"),
            "timesteps": InputPort("timesteps", data_type="tensor", optional=True, description="Массив таймстепов"),
            "condition": InputPort("condition", data_type="any", optional=True, description="Условия (тензор или словарь)"),
            "uncond": InputPort("uncond", data_type="any", optional=True, description="Null condition for CFG"),
            "latents": OutputPort("latents", description="Финальные латенты"),
        }
    
    def process(self, **port_inputs: Any) -> Dict[str, Any]:
        """Выполнить цикл деноизинга.
        
        На каждой итерации передаёт во внутренний граф:
            latents       — текущие латенты (carry variable)
            timestep      — текущий таймстеп
            next_timestep — следующий таймстеп
            condition     — conditioning (постоянный между итерациями)
            + любые доп. входы из port_inputs
        """
        import torch
        from tqdm.auto import tqdm
        from .executor import GraphExecutor
        
        latents = port_inputs.get("initial_latents")
        timesteps = port_inputs.get("timesteps")
        condition = port_inputs.get("condition")
        
        # Determine device from latents
        device = latents.device if latents is not None and hasattr(latents, 'device') else torch.device("cpu")
        
        # Ensure we have a non-empty timestep schedule (avoid 0 iterations → raw noise output)
        if timesteps is None or (hasattr(timesteps, "numel") and timesteps.numel() == 0) or (hasattr(timesteps, "__len__") and len(timesteps) == 0):
            # Match diffusers set_timesteps (leading / linspace) + steps_offset
            num_steps = self.num_iterations
            T = self.num_train_timesteps
            if self.timestep_spacing == "leading":
                step_ratio = T // num_steps
                timesteps = torch.arange(0, num_steps, device=device).long() * step_ratio
                timesteps = timesteps.flip(0)
            else:
                timesteps = torch.linspace(T - 1, 0, num_steps, device=device).long()
            timesteps = (timesteps + self.steps_offset).clamp(0, T - 1)
        elif not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps, device=device)
        else:
            timesteps = timesteps.to(device)
        
        executor = GraphExecutor(no_grad=True)
        
        iterator = enumerate(timesteps)
        if self.show_progress:
            iterator = tqdm(iterator, total=len(timesteps), desc="Sampling")
        
        for i, t in iterator:
            if i + 1 < len(timesteps):
                next_t = timesteps[i + 1]
            else:
                next_t = torch.tensor(0, device=device)
            
            # Ensure timesteps are proper tensors
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=device)
            t = t.unsqueeze(0) if t.dim() == 0 else t
            next_t = next_t.unsqueeze(0) if next_t.dim() == 0 else next_t
            
            # Backbone: long for DDIM/UNet time_embed; float for flow/SD3 (scheduler timesteps)
            step_inputs = {
                "latents": latents,
                "timestep": t.to(dtype=torch.float32) if t.dtype in (torch.float32, torch.float16, torch.float64) else (t if t.dtype in (torch.long, torch.int64) else t.long()),
                "next_timestep": next_t.to(dtype=torch.float32) if next_t.dtype in (torch.float32, torch.float16, torch.float64) else next_t,
                "condition": condition,
            }
            if i == 0:
                step_inputs["num_steps"] = len(timesteps)
            # Forward extra inputs (e.g. control_image for ControlNet) to inner graph every step
            for k, v in port_inputs.items():
                if k not in ("initial_latents", "timesteps", "condition"):
                    step_inputs[k] = v
            
            step_outputs = executor.execute(self.graph, **step_inputs)
            
            # Update carried variables (must get new latents from step or loop is no-op → noisy image)
            # Do not use "or": tensors trigger "Boolean value of Tensor with more than one value is ambiguous"
            next_lat = step_outputs.get("next_latents")
            if next_lat is None:
                next_lat = step_outputs.get("latents")
            if next_lat is None:
                next_lat = step_outputs.get("output")
            if next_lat is not None:
                latents = next_lat.detach().clone() if hasattr(next_lat, "detach") else next_lat
            else:
                import logging
                logging.getLogger(__name__).warning(
                    "Denoise step returned no latents/next_latents/output; loop may produce noise. Keys: %s",
                    list(step_outputs.keys()),
                )
        
        return {"latents": latents, "output": latents}
    
    def _forward_impl(self, **kwargs) -> Any:
        return self.process(**kwargs)
    
    def __repr__(self) -> str:
        n_nodes = len(self.graph.nodes) if self._inner_graph else 0
        return f"<LoopSubGraph '{self.block_id}' iters={self.num_iterations} inner_nodes={n_nodes}>"
