"""Generation tab layout builder."""
from __future__ import annotations

from typing import Dict, Any, Optional


def build_inputs_from_schema(schema: Dict[str, Any]):
    """Build Gradio input components from plugin UI schema.
    
    Args:
        schema: Plugin's get_ui_schema() result
        
    Returns:
        Dict of Gradio components keyed by name
    """
    import gradio as gr
    
    components = {}
    
    for field in schema.get("inputs", []):
        ftype = field["type"]
        name = field["name"]
        label = field.get("label", name)
        optional = field.get("optional", False)
        
        if ftype == "text":
            components[name] = gr.Textbox(
                label=label,
                placeholder=field.get("placeholder", ""),
                lines=2 if "prompt" in name else 1,
            )
        elif ftype == "image":
            components[name] = gr.Image(label=label, type="pil")
        elif ftype == "audio":
            components[name] = gr.Audio(label=label)
        elif ftype == "file":
            components[name] = gr.File(label=label)
        elif ftype == "dataframe":
            components[name] = gr.Dataframe(label=label)
    
    return components


def build_outputs_from_schema(schema: Dict[str, Any]):
    """Build Gradio output components from plugin UI schema."""
    import gradio as gr
    
    components = {}
    
    for field in schema.get("outputs", []):
        ftype = field["type"]
        name = field["name"]
        label = field.get("label", name)
        
        if ftype == "image":
            components[name] = gr.Image(label=label)
        elif ftype == "gallery":
            components[name] = gr.Gallery(label=label)
        elif ftype == "audio":
            components[name] = gr.Audio(label=label)
        elif ftype == "video":
            components[name] = gr.Video(label=label)
        elif ftype == "3d":
            components[name] = gr.Model3D(label=label)
        elif ftype == "text":
            components[name] = gr.Textbox(label=label)
        elif ftype == "plot":
            components[name] = gr.Plot(label=label)
        elif ftype == "file":
            components[name] = gr.File(label=label)
    
    return components


def build_advanced_from_schema(schema: Dict[str, Any]):
    """Build Gradio advanced settings components."""
    import gradio as gr
    
    components = {}
    
    for field in schema.get("advanced", []):
        ftype = field["type"]
        name = field["name"]
        label = field.get("label", name)
        
        if ftype == "slider":
            components[name] = gr.Slider(
                label=label,
                minimum=field.get("min", 0),
                maximum=field.get("max", 100),
                value=field.get("default", 50),
                step=field.get("step", 1),
            )
        elif ftype == "dropdown":
            components[name] = gr.Dropdown(
                label=label,
                choices=field.get("options", []),
                value=field.get("options", [""])[0] if field.get("options") else None,
            )
        elif ftype == "number":
            components[name] = gr.Number(
                label=label,
                value=field.get("default", 0),
            )
        elif ftype == "checkbox":
            components[name] = gr.Checkbox(
                label=label,
                value=field.get("default", False),
            )
    
    return components
