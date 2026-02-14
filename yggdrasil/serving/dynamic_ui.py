"""Dynamic Gradio UI — auto-generates interface from ComputeGraph I/O.

Key principle: ANY graph automatically gets a working UI.

The UI introspects the graph's input/output ports and creates appropriate
Gradio components for each port type. No hardcoded fields.

Usage:
    from yggdrasil.serving.dynamic_ui import DynamicUI
    
    graph = ComputeGraph.from_template("sd15_txt2img")
    ui = DynamicUI(graph)
    ui.launch()
    
    # Or with Pipeline
    pipe = Pipeline.from_template("flux2_txt2img")
    ui = DynamicUI.from_pipeline(pipe)
    ui.launch()
"""
from __future__ import annotations

import json
import time
import logging
import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


# ==================== PORT → GRADIO COMPONENT MAPPING ====================

# Maps port data_type to Gradio component factory
_PORT_TYPE_MAP = {
    "text": "textbox",
    "str": "textbox",
    "string": "textbox",
    "prompt": "textbox",
    "image": "image",
    "audio": "audio",
    "video": "video",
    "tensor": "textbox",  # raw tensor specs as JSON
    "int": "number",
    "float": "slider",
    "bool": "checkbox",
    "dict": "json",
    "any": "textbox",
}

# Maps port name patterns to more specific components
_PORT_NAME_HINTS = {
    "prompt": ("textbox", {"lines": 3, "placeholder": "Enter your prompt..."}),
    "negative_prompt": ("textbox", {"lines": 2, "placeholder": "Negative prompt..."}),
    "negative": ("textbox", {"lines": 2, "placeholder": "Negative prompt..."}),
    "image": ("image", {"type": "pil"}),
    "input_image": ("image", {"type": "pil"}),
    "mask": ("image", {"type": "pil"}),
    "audio": ("audio", {}),
    "video": ("video", {}),
    "guidance_scale": ("slider", {"minimum": 0, "maximum": 30, "value": 7.5, "step": 0.5}),
    "cfg_scale": ("slider", {"minimum": 0, "maximum": 30, "value": 7.5, "step": 0.5}),
    "num_steps": ("slider", {"minimum": 1, "maximum": 150, "value": 28, "step": 1}),
    "steps": ("slider", {"minimum": 1, "maximum": 150, "value": 28, "step": 1}),
    "width": ("slider", {"minimum": 128, "maximum": 2048, "value": 512, "step": 64}),
    "height": ("slider", {"minimum": 128, "maximum": 2048, "value": 512, "step": 64}),
    "seed": ("number", {"value": -1, "precision": 0}),
    "num_frames": ("slider", {"minimum": 1, "maximum": 128, "value": 16, "step": 1}),
    "strength": ("slider", {"minimum": 0, "maximum": 1, "value": 0.75, "step": 0.05}),
    "batch_size": ("slider", {"minimum": 1, "maximum": 8, "value": 1, "step": 1}),
}


def _infer_component(port_name: str, port_data_type: str = "any") -> Tuple[str, dict]:
    """Infer the best Gradio component type and kwargs from port info."""
    # First check name hints (more specific)
    name_lower = port_name.lower()
    for pattern, (comp_type, kwargs) in _PORT_NAME_HINTS.items():
        if pattern in name_lower:
            return comp_type, {**kwargs, "label": port_name}
    
    # Then check data type
    comp_type = _PORT_TYPE_MAP.get(port_data_type, "textbox")
    return comp_type, {"label": port_name}


def _create_gradio_component(comp_type: str, kwargs: dict):
    """Create a Gradio component from type string and kwargs."""
    import gradio as gr
    
    factories = {
        "textbox": gr.Textbox,
        "number": gr.Number,
        "slider": gr.Slider,
        "checkbox": gr.Checkbox,
        "image": gr.Image,
        "audio": gr.Audio,
        "video": gr.Video,
        "json": gr.JSON,
        "dropdown": gr.Dropdown,
    }
    
    factory = factories.get(comp_type, gr.Textbox)
    return factory(**kwargs)


def _create_output_component(port_name: str, port_data_type: str = "any"):
    """Create an output Gradio component."""
    import gradio as gr
    
    name_lower = port_name.lower()
    
    # Image outputs
    if any(p in name_lower for p in ["image", "output", "result"]):
        return gr.Gallery(label=port_name, columns=2, height=500, object_fit="contain")
    
    # Video outputs
    if "video" in name_lower:
        return gr.Video(label=port_name)
    
    # Audio outputs
    if "audio" in name_lower:
        return gr.Audio(label=port_name)
    
    # Latent / tensor outputs
    if "latent" in name_lower:
        return gr.Textbox(label=f"{port_name} (shape)", interactive=False)
    
    # Default: text
    return gr.Textbox(label=port_name, interactive=False)


def _tensor_to_images(tensor: torch.Tensor) -> List[Image.Image]:
    """Convert output tensor to PIL images."""
    if tensor is None:
        return []
    
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
        
        # Handle different tensor shapes
        if tensor.ndim == 3:  # (C, H, W)
            tensor = tensor.unsqueeze(0)
        
        if tensor.ndim != 4:
            return []
        
        images = []
        for i in range(tensor.shape[0]):
            img = tensor[i]
            # Normalize from [-1, 1] to [0, 1]
            if img.min() < 0:
                img = (img / 2 + 0.5).clamp(0, 1)
            elif img.max() > 1:
                img = img.clamp(0, 255).to(torch.uint8)
            else:
                img = img.clamp(0, 1)
            
            img = (img * 255).to(torch.uint8).numpy()
            if img.shape[0] in (1, 3, 4):
                img = img.transpose(1, 2, 0)
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
            images.append(Image.fromarray(img))
        
        return images
    
    return []


# ==================== DYNAMIC UI CLASS ====================

class DynamicUI:
    """Dynamically generate a Gradio UI from a ComputeGraph or Pipeline.
    
    Introspects graph I/O ports and creates appropriate UI components.
    Works with any graph — no hardcoded fields.
    
    Features:
    - Auto-generates input components from graph input ports
    - Auto-generates output components from graph output ports
    - Parameter panel from graph metadata (guidance_scale, num_steps, etc.)
    - Graph visualization tab
    - Block registry browser
    - Live graph editing
    """
    
    def __init__(
        self,
        graph: Optional["ComputeGraph"] = None,
        pipeline: Optional["Pipeline"] = None,
        title: str = "YggDrasil",
        theme: str = "soft",
    ):
        self.graph = graph
        self.pipeline = pipeline
        self.title = title
        self.theme = theme
        self._custom_tabs: List[Tuple[str, Callable]] = []
    
    @classmethod
    def from_pipeline(cls, pipeline: "Pipeline", **kwargs) -> DynamicUI:
        """Create UI from a Pipeline instance."""
        return cls(graph=pipeline.graph, pipeline=pipeline, **kwargs)
    
    @classmethod
    def from_template(cls, template_name: str, **kwargs) -> DynamicUI:
        """Create UI from a template name."""
        from ..core.graph.graph import ComputeGraph
        graph = ComputeGraph.from_template(template_name)
        return cls(graph=graph, **kwargs)
    
    def add_tab(self, name: str, builder: Callable):
        """Add a custom tab to the UI."""
        self._custom_tabs.append((name, builder))
        return self
    
    def _get_graph_inputs(self) -> Dict[str, Dict[str, str]]:
        """Introspect graph to determine user-facing inputs.
        
        Uses graph_inputs (exposed ports) as the source of truth.
        Falls back to node-level port scanning only if no graph_inputs are defined.
        No hardcoded fields — a molecular graph without "prompt" won't show one.
        """
        inputs = {}
        
        if self.graph is None:
            return inputs
        
        # PRIMARY: Use graph's exposed inputs (the real API)
        if hasattr(self.graph, 'graph_inputs') and self.graph.graph_inputs:
            for input_name, targets in self.graph.graph_inputs.items():
                # Skip internal-only inputs (latents, timesteps, etc.)
                if input_name in ("latents",):
                    continue
                
                # Determine type from target block's port declaration
                data_type = "any"
                required = True
                
                for target_node, target_port in targets:
                    block = self.graph.nodes.get(target_node)
                    if block is not None:
                        try:
                            io = block.declare_io()
                            port = io.get(target_port)
                            if port is not None:
                                data_type = getattr(port, 'data_type', 'any')
                                required = not getattr(port, 'optional', False)
                        except Exception:
                            pass
                
                inputs[input_name] = {"type": data_type, "required": required}
        
        # SECONDARY: Scan node ports for additional discoverable inputs
        if not inputs:
            for name, block in self.graph.nodes.items():
                try:
                    io = block.declare_io() if hasattr(block, 'declare_io') else {}
                except Exception:
                    continue
                
                for port_name, port in io.items():
                    if not (hasattr(port, 'direction') and port.direction == "input"):
                        continue
                    
                    data_type = getattr(port, 'data_type', 'any')
                    optional = getattr(port, 'optional', False)
                    
                    if port_name not in inputs:
                        inputs[port_name] = {
                            "type": data_type,
                            "required": not optional,
                        }
        
        # Add standard generation parameters from metadata (not hardcoded)
        meta = getattr(self.graph, 'metadata', {}) or {}
        
        # Only add width/height if metadata suggests image generation
        if meta.get("default_width") or meta.get("default_height"):
            if "width" not in inputs:
                inputs["width"] = {"type": "int", "required": False}
            if "height" not in inputs:
                inputs["height"] = {"type": "int", "required": False}
        
        # Always add seed for reproducibility
        if "seed" not in inputs:
            inputs["seed"] = {"type": "int", "required": False}
        
        return inputs
    
    def _get_graph_outputs(self) -> List[str]:
        """Introspect graph to determine output types.
        
        Uses graph_outputs and metadata to determine what the graph produces.
        """
        if self.graph is None:
            return ["images"]
        
        outputs = []
        
        # Check graph metadata for modality
        meta = getattr(self.graph, 'metadata', {}) or {}
        modality = meta.get("modality", "")
        
        # Check actual graph outputs
        if hasattr(self.graph, 'graph_outputs') and self.graph.graph_outputs:
            for output_name in self.graph.graph_outputs:
                name_lower = output_name.lower()
                if any(k in name_lower for k in ["image", "decoded", "output", "result"]):
                    if "images" not in outputs:
                        outputs.append("images")
                elif "video" in name_lower:
                    if "video" not in outputs:
                        outputs.append("video")
                elif "audio" in name_lower:
                    if "audio" not in outputs:
                        outputs.append("audio")
        
        # Fallback based on modality
        if not outputs:
            if modality == "video":
                outputs = ["video"]
            elif modality == "audio":
                outputs = ["audio"]
            else:
                outputs = ["images"]
        
        return outputs
    
    def build(self) -> "gr.Blocks":
        """Build the full Gradio Blocks interface."""
        import gradio as gr
        
        theme_map = {
            "soft": gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
            "dark": gr.themes.Base(),
            "glass": gr.themes.Glass(),
            "monochrome": gr.themes.Monochrome(),
        }
        
        with gr.Blocks(
            title=f"{self.title} — Universal Diffusion Framework",
            theme=theme_map.get(self.theme, gr.themes.Soft()),
            css="""
            .main-title { text-align: center; margin-bottom: 0.5em; }
            .subtitle { text-align: center; color: #666; margin-bottom: 1.5em; }
            .port-section { border-left: 3px solid #4f46e5; padding-left: 1em; margin: 0.5em 0; }
            """,
        ) as demo:
            
            gr.HTML(f"""
            <h1 class="main-title">{self.title}</h1>
            <p class="subtitle">True Lego Constructor for Diffusion Models</p>
            """)
            
            with gr.Tabs():
                self._build_generate_tab(gr)
                self._build_graph_editor_tab(gr)
                self._build_training_tab(gr)
                self._build_workflow_tab(gr)
                self._build_blocks_tab(gr)
                self._build_deploy_tab(gr)
                
                # Custom tabs
                for tab_name, builder in self._custom_tabs:
                    with gr.Tab(tab_name):
                        builder(gr)
            
            gr.HTML("""
            <div style="text-align: center; margin-top: 2em; color: #888; font-size: 0.85em;">
                YggDrasil — True Lego Constructor for Diffusion Models
            </div>
            """)
        
        return demo
    
    def _build_generate_tab(self, gr):
        """Build the dynamic generation tab."""
        graph_inputs = self._get_graph_inputs()
        graph_outputs = self._get_graph_outputs()
        
        with gr.Tab("Generate"):
            with gr.Row():
                # Input column
                with gr.Column(scale=1):
                    gr.Markdown("### Inputs")
                    
                    # Template selector
                    template_names = self._get_template_names()
                    template_dropdown = gr.Dropdown(
                        label="Pipeline Template",
                        choices=template_names,
                        value=template_names[0] if template_names else None,
                        interactive=True,
                    )
                    
                    # Dynamically create input components
                    input_components = {}
                    
                    for port_name, port_info in graph_inputs.items():
                        comp_type, comp_kwargs = _infer_component(port_name, port_info.get("type", "any"))
                        component = _create_gradio_component(comp_type, comp_kwargs)
                        input_components[port_name] = component
                    
                    # Negative prompt (if text-based)
                    if "prompt" in input_components:
                        neg_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="blurry, low quality",
                            lines=2,
                        )
                        input_components["negative_prompt"] = neg_prompt
                    
                    generate_btn = gr.Button("Generate", variant="primary", size="lg")
                
                # Output column
                with gr.Column(scale=2):
                    gr.Markdown("### Output")
                    
                    output_components = {}
                    
                    if "images" in graph_outputs:
                        output_components["gallery"] = gr.Gallery(
                            label="Results", columns=2, height=600, object_fit="contain"
                        )
                    if "video" in graph_outputs:
                        output_components["video"] = gr.Video(label="Generated Video")
                    if "audio" in graph_outputs:
                        output_components["audio"] = gr.Audio(label="Generated Audio")
                    
                    info_box = gr.Textbox(label="Info", interactive=False, lines=2)
            
            # Wire up generation
            all_inputs = list(input_components.values())
            all_outputs = list(output_components.values()) + [info_box]
            
            def generate_fn(*args):
                # Map positional args back to named inputs
                input_names = list(input_components.keys())
                kwargs = {name: val for name, val in zip(input_names, args)}
                return self._run_generation(kwargs, graph_outputs)
            
            generate_btn.click(fn=generate_fn, inputs=all_inputs, outputs=all_outputs)
    
    def _run_generation(self, inputs: Dict[str, Any], output_types: List[str]):
        """Execute generation with given inputs."""
        try:
            start_time = time.time()
            
            # Build generation kwargs
            gen_kwargs = {}
            
            prompt = inputs.get("prompt", "")
            if prompt:
                gen_kwargs["prompt"] = prompt
            
            neg = inputs.get("negative_prompt", "")
            if neg:
                gen_kwargs["negative_prompt"] = neg
            
            for key in ["guidance_scale", "num_steps", "width", "height", "seed"]:
                val = inputs.get(key)
                if val is not None and val != "":
                    gen_kwargs[key] = float(val) if key == "guidance_scale" else int(val)
            
            # Handle image inputs
            for key in ["input_image", "mask", "image"]:
                img = inputs.get(key)
                if img is not None:
                    if isinstance(img, Image.Image):
                        # Convert PIL to tensor
                        import torchvision.transforms as T
                        gen_kwargs[key] = T.ToTensor()(img).unsqueeze(0) * 2 - 1
                    else:
                        gen_kwargs[key] = img
            
            # Handle seed
            seed = gen_kwargs.pop("seed", -1)
            if seed >= 0:
                gen_kwargs["generator"] = torch.Generator().manual_seed(int(seed))
            
            # Execute
            if self.pipeline is not None:
                result = self.pipeline(**gen_kwargs)
                output_tensor = result.images if hasattr(result, 'images') else result
            elif self.graph is not None:
                from ..pipeline import Pipeline
                pipe = Pipeline.from_graph(self.graph)
                result = pipe(**gen_kwargs)
                output_tensor = result.images if hasattr(result, 'images') else result
            else:
                return [], "No graph or pipeline loaded"
            
            elapsed = time.time() - start_time
            
            # Convert output
            results = []
            if "images" in output_types:
                if isinstance(output_tensor, torch.Tensor):
                    images = _tensor_to_images(output_tensor)
                elif isinstance(output_tensor, list):
                    images = output_tensor
                else:
                    images = []
                results.append(images)
            
            if "video" in output_types:
                results.append(None)  # Placeholder
            
            if "audio" in output_types:
                results.append(None)  # Placeholder
            
            info = f"Time: {elapsed:.1f}s | Seed: {seed}"
            results.append(info)
            
            return results
            
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            n_outputs = len(output_types)
            return [None] * n_outputs + [f"Error: {e}"]
    
    def _build_graph_editor_tab(self, gr):
        """Build the interactive graph editor tab."""
        with gr.Tab("Graph Editor"):
            gr.Markdown("### Visual Pipeline Builder")
            gr.Markdown("Build custom pipelines by combining blocks like Lego.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    template_dd = gr.Dropdown(
                        label="Template",
                        choices=self._get_template_names(),
                    )
                    load_btn = gr.Button("Load Template", variant="primary")
                    
                    gr.Markdown("#### Add Node")
                    node_name = gr.Textbox(label="Name", placeholder="my_block")
                    
                    # Get all registered block types
                    block_types = self._get_block_types()
                    node_type = gr.Dropdown(
                        label="Block Type",
                        choices=block_types,
                        allow_custom_value=True,
                    )
                    node_config = gr.Code(label="Config (JSON)", language="json", value="{}", lines=4)
                    
                    with gr.Row():
                        add_btn = gr.Button("Add")
                        replace_btn = gr.Button("Replace")
                        remove_btn = gr.Button("Remove", variant="stop")
                    
                    gr.Markdown("#### Connect Ports")
                    src_spec = gr.Textbox(label="Source", placeholder="backbone.output")
                    dst_spec = gr.Textbox(label="Destination", placeholder="guidance.model_output")
                    connect_btn = gr.Button("Connect")
                    
                    gr.Markdown("#### LoRA")
                    lora_node = gr.Textbox(label="Target Node", placeholder="backbone")
                    lora_rank = gr.Slider(1, 128, value=16, step=1, label="Rank")
                    lora_modules = gr.Textbox(
                        label="Target Modules",
                        value="to_q, to_k, to_v, to_out.0",
                    )
                    apply_lora_btn = gr.Button("Apply LoRA")
                
                with gr.Column(scale=2):
                    graph_viz = gr.Code(label="Graph (Mermaid)", language="markdown", lines=15)
                    graph_nodes = gr.Textbox(label="Nodes", interactive=False, lines=8)
                    graph_edges = gr.Textbox(label="Edges", interactive=False, lines=8)
            
            graph_state = gr.State(value=self.graph)
            
            def load_template_fn(name):
                try:
                    from ..core.graph.graph import ComputeGraph
                    g = ComputeGraph.from_template(name)
                    viz = g.visualize() if hasattr(g, 'visualize') else str(g)
                    nodes = "\n".join(f"  {n}: {getattr(b, 'block_type', type(b).__name__)}" for n, b in g.nodes.items())
                    edges = "\n".join(f"  {e}" for e in (g.edges if hasattr(g, 'edges') else []))
                    return g, viz, nodes, edges
                except Exception as e:
                    return None, "", f"Error: {e}", ""
            
            def add_node_fn(g, name, btype, cfg):
                if g is None:
                    return g, "No graph", "", ""
                try:
                    from ..core.block.builder import BlockBuilder
                    config = json.loads(cfg) if cfg.strip() else {}
                    config["type"] = btype
                    block = BlockBuilder.build(config)
                    g.add_node(name, block)
                    viz = g.visualize() if hasattr(g, 'visualize') else str(g)
                    nodes = "\n".join(f"  {n}: {getattr(b, 'block_type', type(b).__name__)}" for n, b in g.nodes.items())
                    edges = "\n".join(f"  {e}" for e in (g.edges if hasattr(g, 'edges') else []))
                    return g, viz, nodes, edges
                except Exception as e:
                    return g, "", f"Error: {e}", ""
            
            def remove_node_fn(g, name):
                if g is None:
                    return g, "No graph", "", ""
                try:
                    g.remove_node(name)
                    viz = g.visualize() if hasattr(g, 'visualize') else str(g)
                    nodes = "\n".join(f"  {n}: {getattr(b, 'block_type', type(b).__name__)}" for n, b in g.nodes.items())
                    edges = "\n".join(f"  {e}" for e in (g.edges if hasattr(g, 'edges') else []))
                    return g, viz, nodes, edges
                except Exception as e:
                    return g, "", f"Error: {e}", ""
            
            def connect_fn(g, src, dst):
                if g is None:
                    return g, "No graph", "", ""
                try:
                    sn, sp = src.split(".", 1)
                    dn, dp = dst.split(".", 1)
                    g.connect(sn, sp, dn, dp)
                    viz = g.visualize() if hasattr(g, 'visualize') else str(g)
                    nodes = "\n".join(f"  {n}: {getattr(b, 'block_type', type(b).__name__)}" for n, b in g.nodes.items())
                    edges = "\n".join(f"  {e}" for e in (g.edges if hasattr(g, 'edges') else []))
                    return g, viz, nodes, edges
                except Exception as e:
                    return g, "", f"Error: {e}", ""
            
            def apply_lora_fn(g, target, rank, modules):
                if g is None:
                    return "No graph loaded"
                try:
                    from ..blocks.adapters.lora import apply_lora
                    mods = [m.strip() for m in modules.split(",")]
                    apply_lora(g, {target: {"rank": int(rank), "target_modules": mods}})
                    return f"LoRA applied to {target} (rank={rank})"
                except Exception as e:
                    return f"Error: {e}"
            
            load_btn.click(fn=load_template_fn, inputs=[template_dd], outputs=[graph_state, graph_viz, graph_nodes, graph_edges])
            add_btn.click(fn=add_node_fn, inputs=[graph_state, node_name, node_type, node_config], outputs=[graph_state, graph_viz, graph_nodes, graph_edges])
            remove_btn.click(fn=remove_node_fn, inputs=[graph_state, node_name], outputs=[graph_state, graph_viz, graph_nodes, graph_edges])
            connect_btn.click(fn=connect_fn, inputs=[graph_state, src_spec, dst_spec], outputs=[graph_state, graph_viz, graph_nodes, graph_edges])
            apply_lora_btn.click(fn=apply_lora_fn, inputs=[graph_state, lora_node, lora_rank, lora_modules], outputs=[gr.Textbox(visible=False)])
    
    def _build_training_tab(self, gr):
        """Build the training tab."""
        with gr.Tab("Train"):
            gr.Markdown("### Train Any Component")
            gr.Markdown("Select which graph nodes to train — everything else is frozen.")
            
            with gr.Row():
                with gr.Column():
                    train_template = gr.Dropdown(
                        label="Template", choices=self._get_template_names(),
                    )
                    train_nodes = gr.Textbox(
                        label="Train Nodes (comma-separated)",
                        placeholder="backbone, my_adapter",
                        value="backbone",
                    )
                    train_dataset = gr.Textbox(label="Dataset Path", placeholder="/path/to/data/")
                    
                    gr.Markdown("#### Per-Node Learning Rates (JSON)")
                    node_lr_input = gr.Code(
                        label="Node LR",
                        language="json",
                        value='{"backbone": 1e-5}',
                        lines=3,
                    )
                    
                    gr.Markdown("#### Training Schedule (JSON)")
                    schedule_input = gr.Code(
                        label="Schedule",
                        language="json",
                        value='[\n  {"epoch": 0, "freeze": ["backbone"]},\n  {"epoch": 5, "unfreeze": ["backbone"], "set_lr": {"backbone": 1e-6}}\n]',
                        lines=6,
                    )
                
                with gr.Column():
                    train_epochs = gr.Slider(1, 1000, value=10, step=1, label="Epochs")
                    train_batch = gr.Slider(1, 64, value=1, step=1, label="Batch Size")
                    train_lr = gr.Number(label="Base Learning Rate", value=1e-4)
                    train_optimizer = gr.Dropdown(
                        label="Optimizer",
                        choices=["adamw", "adam", "sgd"],
                        value="adamw",
                    )
                    train_scheduler = gr.Dropdown(
                        label="LR Scheduler",
                        choices=["cosine", "cosine_warmup", "linear", "constant"],
                        value="cosine",
                    )
                    train_precision = gr.Dropdown(
                        label="Mixed Precision",
                        choices=["no", "fp16", "bf16"],
                        value="no",
                    )
                    train_grad_accum = gr.Slider(1, 64, value=1, step=1, label="Gradient Accumulation")
                    train_ema = gr.Checkbox(label="Use EMA", value=False)
                    train_save_every = gr.Slider(100, 5000, value=500, step=100, label="Save Every N Steps")
            
            train_btn = gr.Button("Start Training", variant="primary", size="lg")
            train_status = gr.Textbox(label="Status", interactive=False, lines=3)
            
            def start_training(template, nodes_str, dataset, node_lr_json, schedule_json,
                             epochs, batch, lr, optimizer, scheduler, precision, grad_accum, ema, save_every):
                try:
                    from ..core.graph.graph import ComputeGraph
                    from ..training.graph_trainer import GraphTrainer, GraphTrainingConfig
                    from ..training.data import ImageFolderSource
                    
                    nodes = [n.strip() for n in nodes_str.split(",") if n.strip()]
                    node_lr = json.loads(node_lr_json) if node_lr_json.strip() else {}
                    schedule = json.loads(schedule_json) if schedule_json.strip() else []
                    
                    config = GraphTrainingConfig(
                        num_epochs=int(epochs),
                        batch_size=int(batch),
                        learning_rate=float(lr),
                        optimizer=optimizer,
                        lr_scheduler=scheduler,
                        mixed_precision=precision,
                        gradient_accumulation_steps=int(grad_accum),
                        use_ema=ema,
                        save_every=int(save_every),
                        node_lr=node_lr,
                        schedule=schedule,
                    )
                    
                    g = ComputeGraph.from_template(template)
                    trainer = GraphTrainer(graph=g, train_nodes=nodes, config=config)
                    ds = ImageFolderSource(dataset)
                    
                    import threading
                    def run():
                        try:
                            trainer.train(ds)
                        except Exception as e:
                            logger.error(f"Training error: {e}")
                    
                    threading.Thread(target=run, daemon=True).start()
                    return f"Training started | Nodes: {nodes} | Epochs: {epochs} | LR: {node_lr or lr}"
                except Exception as e:
                    return f"Error: {e}"
            
            train_btn.click(
                fn=start_training,
                inputs=[train_template, train_nodes, train_dataset, node_lr_input, schedule_input,
                        train_epochs, train_batch, train_lr, train_optimizer, train_scheduler,
                        train_precision, train_grad_accum, train_ema, train_save_every],
                outputs=[train_status],
            )
    
    def _build_blocks_tab(self, gr):
        """Build the block registry browser tab."""
        with gr.Tab("Blocks"):
            gr.Markdown("### Registered Blocks (Lego Bricks)")
            gr.Markdown("All available blocks in the framework. Use these to build custom pipelines.")
            
            def get_blocks_md():
                try:
                    from ..core.block.registry import list_blocks
                    blocks = list_blocks()
                    if not blocks:
                        return "No blocks registered"
                    
                    categories = {}
                    for key, cls in sorted(blocks.items()):
                        cat = key.split("/")[0] if "/" in key else "other"
                        categories.setdefault(cat, []).append((key, cls))
                    
                    lines = []
                    for cat, items in sorted(categories.items()):
                        lines.append(f"\n### {cat.upper()}")
                        for key, cls in items:
                            doc = (cls.__doc__ or "").split("\n")[0].strip()[:80]
                            ports_str = ""
                            try:
                                io = cls.declare_io()
                                if io:
                                    ins = [p.name for p in io.values() if hasattr(p, 'direction') and p.direction == "input"]
                                    outs = [p.name for p in io.values() if hasattr(p, 'direction') and p.direction == "output"]
                                    if ins or outs:
                                        ports_str = f"\n  - In: `{ins}` | Out: `{outs}`"
                            except Exception:
                                pass
                            lines.append(f"- **`{key}`** — {doc}{ports_str}")
                    
                    return "\n".join(lines)
                except Exception as e:
                    return f"Error loading blocks: {e}"
            
            blocks_md = gr.Markdown(value=get_blocks_md)
            gr.Button("Refresh").click(fn=get_blocks_md, outputs=[blocks_md])
    
    def _build_workflow_tab(self, gr):
        """Build the workflow import/export tab."""
        with gr.Tab("Workflows"):
            gr.Markdown("### Workflow Manager")
            gr.Markdown("Save and load complete workflows (graph + parameters) — like ComfyUI.")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Export Workflow")
                    export_format = gr.Dropdown(
                        label="Format", choices=["yaml", "json"], value="yaml"
                    )
                    export_btn = gr.Button("Export Current Workflow", variant="primary")
                    export_output = gr.Code(label="Workflow Content", language="yaml", lines=20)
                    export_download = gr.File(label="Download", interactive=False)
                
                with gr.Column():
                    gr.Markdown("#### Import Workflow")
                    import_file = gr.File(label="Upload Workflow (.yaml / .json)")
                    import_btn = gr.Button("Load Workflow", variant="primary")
                    import_status = gr.Textbox(label="Status", interactive=False, lines=3)
            
            graph_state = gr.State(value=self.graph)
            
            def export_fn(fmt):
                if self.graph is None:
                    return "No graph loaded", None
                try:
                    import tempfile
                    ext = ".yaml" if fmt == "yaml" else ".json"
                    path = tempfile.mktemp(suffix=ext)
                    self.graph.to_workflow(path)
                    with open(path) as f:
                        content = f.read()
                    return content, path
                except Exception as e:
                    return f"Error: {e}", None
            
            def import_fn(file_obj):
                if file_obj is None:
                    return "No file selected"
                try:
                    from ..core.graph.graph import ComputeGraph
                    graph, params = ComputeGraph.from_workflow(file_obj.name)
                    self.graph = graph
                    return (
                        f"Loaded workflow '{graph.name}'\n"
                        f"Nodes: {len(graph.nodes)} | Edges: {len(graph.edges)}\n"
                        f"Parameters: {list(params.keys())}"
                    )
                except Exception as e:
                    return f"Error: {e}"
            
            export_btn.click(fn=export_fn, inputs=[export_format], outputs=[export_output, export_download])
            import_btn.click(fn=import_fn, inputs=[import_file], outputs=[import_status])
    
    def _build_deploy_tab(self, gr):
        """Build the deployment tab."""
        with gr.Tab("Deploy"):
            gr.Markdown("### Deploy Pipeline")
            
            with gr.Row():
                with gr.Column():
                    deploy_target = gr.Dropdown(
                        label="Target",
                        choices=["Local Server (FastAPI)", "RunPod Serverless", "Vast.ai GPU"],
                        value="Local Server (FastAPI)",
                    )
                    deploy_template = gr.Dropdown(
                        label="Pipeline", choices=self._get_template_names(),
                    )
                    deploy_port = gr.Number(label="Port", value=8000, precision=0)
                    deploy_workers = gr.Slider(1, 8, value=1, step=1, label="Workers")
                    deploy_api_key = gr.Textbox(label="API Key (cloud)", type="password")
                    deploy_gpu = gr.Dropdown(
                        label="GPU Type (cloud)",
                        choices=["RTX_4090", "A100_40GB", "A100_80GB", "H100"],
                        value="RTX_4090",
                    )
                
                with gr.Column():
                    deploy_btn = gr.Button("Deploy", variant="primary", size="lg")
                    deploy_status = gr.Textbox(label="Status", interactive=False, lines=5)
                    
                    gr.Markdown("#### API Endpoint Code")
                    deploy_code = gr.Code(
                        label="Client Code",
                        language="python",
                        value='''import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "a beautiful landscape",
        "num_steps": 28,
        "guidance_scale": 7.5,
    }
)
images = response.json()["images"]
''',
                        lines=12,
                    )
            
            def deploy_fn(target, template, port, workers, api_key, gpu):
                return f"Deployment configured: {target}\nTemplate: {template}\nPort: {port}\nWorkers: {workers}"
            
            deploy_btn.click(
                fn=deploy_fn,
                inputs=[deploy_target, deploy_template, deploy_port, deploy_workers, deploy_api_key, deploy_gpu],
                outputs=[deploy_status],
            )
    
    def _get_template_names(self) -> List[str]:
        try:
            from ..core.graph.templates import list_templates
            return list_templates()
        except Exception:
            return ["sd15_txt2img", "sdxl_txt2img"]
    
    def _get_block_types(self) -> List[str]:
        try:
            from ..core.block.registry import list_blocks
            return sorted(list_blocks().keys())
        except Exception:
            return []
    
    def launch(self, **kwargs):
        """Build and launch the UI."""
        demo = self.build()
        demo.launch(**kwargs)
        return demo
