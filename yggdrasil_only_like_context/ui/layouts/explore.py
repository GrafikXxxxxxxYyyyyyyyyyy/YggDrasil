"""Block graph exploration tab layout."""
from __future__ import annotations


def build_explore_tab():
    """Build the model exploration tab."""
    import gradio as gr
    
    components = {}
    
    with gr.Column():
        components["model_info"] = gr.JSON(
            label="Current Model Structure",
        )
        
        components["block_list"] = gr.Dataframe(
            label="Registered Blocks",
            headers=["Type", "Category", "Class"],
        )
        
        components["graph_html"] = gr.HTML(
            label="Model Graph",
        )
        
        with gr.Row():
            components["refresh_btn"] = gr.Button("Refresh")
            components["export_btn"] = gr.Button("Export Config")
        
        components["exported_config"] = gr.Code(
            label="YAML Config",
            language="yaml",
        )
    
    return components


def get_model_structure(model) -> dict:
    """Extract model structure for display."""
    if model is None:
        return {"status": "No model loaded"}
    
    structure = {
        "block_id": getattr(model, "block_id", "unknown"),
        "block_type": getattr(model, "block_type", "unknown"),
        "nodes": {},
    }
    nodes = getattr(model, "_graph", None) and getattr(model._graph, "nodes", None) or getattr(model, "_slot_children", {})
    if hasattr(nodes, "items"):
        for node_name, block in nodes.items():
            if isinstance(block, list):
                structure["nodes"][node_name] = [
                    {"type": getattr(c, "block_type", type(c).__name__), "id": getattr(c, "block_id", "?")}
                    for c in block
                ]
            elif block is not None:
                structure["nodes"][node_name] = {
                    "type": getattr(block, "block_type", type(block).__name__),
                    "id": getattr(block, "block_id", "?"),
                }
    return structure


def get_registered_blocks_df() -> list:
    """Get registered blocks as a list of rows for Dataframe."""
    from yggdrasil.core.block.registry import BlockRegistry
    
    rows = []
    for key, cls in sorted(BlockRegistry.list_blocks().items()):
        category = key.split("/")[0] if "/" in key else "other"
        rows.append([key, category, cls.__name__])
    
    return rows
