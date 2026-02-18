"""Training tab layout builder."""
from __future__ import annotations


def build_training_tab():
    """Build the training configuration tab."""
    import gradio as gr
    
    components = {}
    
    with gr.Column():
        components["train_mode"] = gr.Dropdown(
            label="Training Mode",
            choices=["full", "adapter", "finetune"],
            value="adapter",
        )
        
        with gr.Row():
            components["num_epochs"] = gr.Number(label="Epochs", value=100)
            components["batch_size"] = gr.Number(label="Batch Size", value=1)
            components["learning_rate"] = gr.Number(label="Learning Rate", value=1e-4)
        
        with gr.Row():
            components["optimizer"] = gr.Dropdown(
                label="Optimizer",
                choices=["adamw", "adam", "sgd"],
                value="adamw",
            )
            components["lr_scheduler"] = gr.Dropdown(
                label="LR Scheduler",
                choices=["constant", "cosine", "linear", "cosine_warmup"],
                value="cosine",
            )
        
        with gr.Row():
            components["mixed_precision"] = gr.Dropdown(
                label="Mixed Precision",
                choices=["no", "fp16", "bf16"],
                value="fp16",
            )
            components["gradient_accumulation"] = gr.Number(
                label="Gradient Accumulation Steps",
                value=4,
            )
        
        with gr.Accordion("Adapter Settings", open=False):
            components["adapter_type"] = gr.Dropdown(
                label="Adapter Type",
                choices=["lora", "dora", "controlnet", "ip_adapter"],
                value="lora",
            )
            components["lora_rank"] = gr.Slider(
                label="LoRA Rank",
                minimum=1, maximum=128, value=4, step=1,
            )
            components["lora_alpha"] = gr.Slider(
                label="LoRA Alpha",
                minimum=0.1, maximum=4.0, value=1.0, step=0.1,
            )
        
        with gr.Row():
            components["data_path"] = gr.Textbox(
                label="Dataset Path",
                placeholder="/path/to/dataset",
            )
            components["save_dir"] = gr.Textbox(
                label="Checkpoint Directory",
                value="checkpoints",
            )
        
        with gr.Row():
            components["start_btn"] = gr.Button("Start Training", variant="primary")
            components["stop_btn"] = gr.Button("Stop Training", variant="stop")
        
        components["training_log"] = gr.Textbox(
            label="Training Log",
            lines=10,
            interactive=False,
        )
        components["loss_plot"] = gr.Plot(label="Loss Curve")
    
    return components
