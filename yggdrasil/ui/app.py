"""YggDrasil Gradio App -- dynamic UI for any diffusion modality.

Usage:
    from yggdrasil.ui import launch
    launch()  # Auto-discovers plugins and creates UI
    
    # Or with a specific model:
    from yggdrasil.assemblers import ModelAssembler
    model = ModelAssembler.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
    launch(model=model)
"""
from __future__ import annotations

import torch
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path


def create_app(
    model=None,
    plugin_name: Optional[str] = None,
    title: str = "YggDrasil Diffusion Studio",
    share: bool = False,
):
    """Create the Gradio application.
    
    Args:
        model: Pre-loaded ModularDiffusionModel (optional)
        plugin_name: Default plugin to use (optional)
        title: App title
        share: Whether to create a public link
        
    Returns:
        Gradio Blocks app
    """
    import gradio as gr
    
    from .components.block_selector import get_available_plugins, get_available_blocks
    from .layouts.generate import (
        build_inputs_from_schema,
        build_outputs_from_schema,
        build_advanced_from_schema,
    )
    from .layouts.train import build_training_tab
    from .layouts.explore import build_explore_tab, get_model_structure, get_registered_blocks_df
    
    # State
    app_state = {
        "model": model,
        "sampler": None,
        "current_plugin": plugin_name,
    }
    
    # Get plugins
    plugins = get_available_plugins()
    plugin_names = [p["name"] for p in plugins]
    if not plugin_names:
        plugin_names = ["image"]
    
    # Get default schema
    default_plugin = plugin_name or (plugin_names[0] if plugin_names else "image")
    
    def get_plugin_schema(name):
        try:
            from yggdrasil.plugins.base import PluginRegistry
            plugin = PluginRegistry.get(name)
            return plugin.get_ui_schema()
        except Exception:
            return {
                "inputs": [{"type": "text", "name": "prompt", "label": "Prompt"}],
                "outputs": [{"type": "image", "name": "result", "label": "Result"}],
                "advanced": [],
            }
    
    def generate_fn(prompt, negative_prompt, num_steps, guidance_scale, width, height, seed, *args):
        """Main generation function."""
        if app_state["model"] is None:
            return None, "No model loaded. Use 'Load Model' first."
        
        model = app_state["model"]
        
        # Build condition
        condition = {"text": prompt}
        if negative_prompt:
            condition["negative_text"] = negative_prompt
        
        # Setup sampler
        from yggdrasil.assemblers.pipeline_assembler import PipelineAssembler
        sampler = PipelineAssembler.for_generation(
            model=model,
            num_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
        )
        
        # Setup seed
        generator = None
        if seed >= 0:
            generator = torch.Generator().manual_seed(int(seed))
        
        # Determine shape
        codec = model._slot_children.get("codec")
        if codec and hasattr(codec, "get_latent_shape"):
            shape = codec.get_latent_shape(1, int(height), int(width))
        else:
            shape = (1, 4, int(height) // 8, int(width) // 8)
        
        # Generate
        try:
            result = sampler.sample(
                condition=condition,
                shape=shape,
                generator=generator,
            )
            
            # Convert to image
            image = result[0].cpu()
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.permute(1, 2, 0).numpy()
            image = (image * 255).clip(0, 255).astype(np.uint8)
            
            return image, f"Generated successfully (shape: {result.shape})"
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def load_model_fn(model_source, model_path, plugin):
        """Load a model from various sources."""
        try:
            from yggdrasil.assemblers.model_assembler import ModelAssembler
            
            if model_source == "HuggingFace":
                model = ModelAssembler.from_pretrained(model_path)
            elif model_source == "YAML Config":
                model = ModelAssembler.from_config(model_path)
            elif model_source == "Plugin Preset":
                model = ModelAssembler.from_plugin(plugin)
            elif model_source == "Recipe":
                model = ModelAssembler.from_recipe(model_path)
            else:
                return "Unknown source"
            
            app_state["model"] = model
            app_state["current_plugin"] = plugin
            
            return f"Model loaded: {getattr(model, 'block_id', 'unknown')}"
        except Exception as e:
            return f"Error loading model: {str(e)}"
    
    # Build the app
    with gr.Blocks(
        title=title,
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 1em; }
        .status-bar { padding: 0.5em; border-radius: 0.5em; }
        """,
    ) as app:
        gr.Markdown(
            f"# {title}\n"
            "Universal diffusion framework -- any modality, any architecture.",
            elem_classes=["main-header"],
        )
        
        with gr.Tabs():
            # === Generation Tab ===
            with gr.Tab("Generate"):
                with gr.Row():
                    # Left panel: Model loading
                    with gr.Column(scale=1):
                        gr.Markdown("### Model")
                        model_source = gr.Dropdown(
                            label="Source",
                            choices=["HuggingFace", "Plugin Preset", "YAML Config", "Recipe"],
                            value="HuggingFace",
                        )
                        model_path = gr.Textbox(
                            label="Model ID / Path",
                            placeholder="stable-diffusion-v1-5/stable-diffusion-v1-5",
                        )
                        plugin_dropdown = gr.Dropdown(
                            label="Plugin",
                            choices=plugin_names,
                            value=default_plugin,
                        )
                        load_btn = gr.Button("Load Model", variant="primary")
                        status_text = gr.Textbox(
                            label="Status",
                            value="No model loaded" if model is None else "Model ready",
                            interactive=False,
                        )
                        
                        load_btn.click(
                            fn=load_model_fn,
                            inputs=[model_source, model_path, plugin_dropdown],
                            outputs=[status_text],
                        )
                    
                    # Center: Generation controls
                    with gr.Column(scale=2):
                        gr.Markdown("### Generation")
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="A beautiful painting of a sunset...",
                            lines=3,
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="blurry, low quality",
                            lines=1,
                        )
                        
                        with gr.Row():
                            num_steps = gr.Slider(
                                label="Steps", minimum=1, maximum=150, value=50, step=1,
                            )
                            guidance_scale = gr.Slider(
                                label="CFG Scale", minimum=1.0, maximum=20.0, value=7.5, step=0.5,
                            )
                        
                        with gr.Row():
                            width = gr.Slider(
                                label="Width", minimum=256, maximum=2048, value=512, step=64,
                            )
                            height = gr.Slider(
                                label="Height", minimum=256, maximum=2048, value=512, step=64,
                            )
                        
                        seed = gr.Number(label="Seed (-1 = random)", value=-1)
                        
                        generate_btn = gr.Button("Generate", variant="primary", size="lg")
                    
                    # Right: Output
                    with gr.Column(scale=2):
                        gr.Markdown("### Output")
                        output_image = gr.Image(label="Result")
                        output_info = gr.Textbox(label="Info", interactive=False)
                
                generate_btn.click(
                    fn=generate_fn,
                    inputs=[prompt, negative_prompt, num_steps, guidance_scale, width, height, seed],
                    outputs=[output_image, output_info],
                )
            
            # === Training Tab ===
            with gr.Tab("Train"):
                train_components = build_training_tab()
            
            # === Explore Tab ===
            with gr.Tab("Explore"):
                explore_components = build_explore_tab()
                
                def refresh_explore():
                    structure = get_model_structure(app_state.get("model"))
                    blocks = get_registered_blocks_df()
                    return structure, blocks
                
                if "refresh_btn" in explore_components:
                    explore_components["refresh_btn"].click(
                        fn=refresh_explore,
                        outputs=[
                            explore_components.get("model_info"),
                            explore_components.get("block_list"),
                        ],
                    )
            
            # === Settings Tab ===
            with gr.Tab("Settings"):
                gr.Markdown("### Device Settings")
                device = gr.Dropdown(
                    label="Device",
                    choices=["auto", "cuda", "mps", "cpu"],
                    value="auto",
                )
                gr.Markdown("### About")
                gr.Markdown(
                    "**YggDrasil** -- Universal Diffusion Framework\n\n"
                    "Build any diffusion model like Lego. "
                    "Any modality, any architecture."
                )
    
    return app


def launch(
    model=None,
    plugin_name: Optional[str] = None,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
    **kwargs,
):
    """Launch the Gradio app.
    
    Args:
        model: Pre-loaded model (optional)
        plugin_name: Default plugin
        server_name: Server hostname
        server_port: Server port
        share: Create public link
    """
    app = create_app(model=model, plugin_name=plugin_name, **kwargs)
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )
