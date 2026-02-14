# Tutorials

## Tutorial 1: Simple Image Generation

```python
from yggdrasil import Pipeline

# One-liner generation (like Diffusers)
pipe = Pipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
result = pipe("a beautiful sunset over mountains", num_steps=28)
result.images[0].save("sunset.png")
```

## Tutorial 2: Swap Any Component (ComfyUI-like)

```python
from yggdrasil.core.graph.graph import ComputeGraph
from yggdrasil import Pipeline

# Load a standard pipeline as a graph
graph = ComputeGraph.from_template("sd15_txt2img")

# See what's inside
print(graph.visualize())
# backbone: backbone/unet2d
# conditioner: conditioner/clip_text
# codec: codec/autoencoder_kl
# solver: solver/euler
# guidance: guidance/cfg

# Replace the solver with DDIM
from yggdrasil.core.block.builder import BlockBuilder
ddim = BlockBuilder.build({"type": "solver/ddim"})
graph.replace_node("solver", ddim)

# Add ControlNet
from yggdrasil.blocks.adapters.lora import apply_lora
controlnet = BlockBuilder.build({"type": "adapter/controlnet", "conditioning_channels": 3})
graph.add_node("controlnet", controlnet)
graph.connect("controlnet", "output", "backbone", "adapter_output")

# Execute the modified graph
pipe = Pipeline.from_graph(graph, device="cuda")
result = pipe(prompt="a house", num_steps=20, guidance_scale=7.5)
```

## Tutorial 3: Train a LoRA Adapter (A1111-like)

```python
from yggdrasil.core.graph.graph import ComputeGraph
from yggdrasil.blocks.adapters.lora import apply_lora, save_lora
from yggdrasil.training import GraphTrainer, GraphTrainingConfig, ImageFolderSource

# 1. Build the graph
graph = ComputeGraph.from_template("sd15_txt2img")
graph.to("cuda", torch.float16)

# 2. Apply LoRA to backbone and text encoder
apply_lora(graph, {
    "backbone": {"rank": 16, "target_modules": ["to_q", "to_k", "to_v", "to_out.0"]},
    "conditioner": {"rank": 8, "target_modules": ["q_proj", "v_proj"]},
})

# 3. Configure training
config = GraphTrainingConfig(
    num_epochs=50,
    batch_size=1,
    learning_rate=1e-4,
    node_lr={"backbone": 1e-4, "conditioner": 5e-5},  # Different LR per component
    optimizer="adamw",
    lr_scheduler="cosine",
    save_every=500,
    mixed_precision="fp16",
    use_ema=True,
)

# 4. Train
dataset = ImageFolderSource("/path/to/my/images", resolution=512)
trainer = GraphTrainer(graph=graph, train_nodes=["backbone", "conditioner"], config=config)
history = trainer.train(dataset)

# 5. Save LoRA weights (tiny file!)
save_lora(graph, "my_style_lora/")

# 6. Later: load and use
from yggdrasil.blocks.adapters.lora import load_lora
graph2 = ComputeGraph.from_template("sd15_txt2img")
load_lora(graph2, "my_style_lora/")
pipe = Pipeline.from_graph(graph2)
result = pipe("in my style, a portrait of a warrior")
```

## Tutorial 4: Progressive Training Schedule

```python
from yggdrasil.training import GraphTrainer, GraphTrainingConfig

# Phase 1 (epoch 0-20): Train only adapter, backbone frozen
# Phase 2 (epoch 20-50): Unfreeze backbone, joint training with lower LR

config = GraphTrainingConfig(
    num_epochs=50,
    learning_rate=1e-4,
    schedule=[
        {"epoch": 0, "freeze": ["backbone"]},
        {"epoch": 20, "unfreeze": ["backbone"]},
        {"epoch": 20, "set_lr": {"backbone": 1e-6}},
    ],
)

trainer = GraphTrainer(
    graph=graph,
    train_nodes=["backbone", "my_adapter"],
    config=config,
)
trainer.train(dataset)
```

## Tutorial 5: Custom Block + Training

```python
import torch.nn as nn
from yggdrasil.core.block.base import AbstractBlock
from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import InputPort, OutputPort

# 1. Define your custom block
@register_block("my/style_transfer")
class StyleTransferBlock(AbstractBlock):
    block_type = "my/style_transfer"
    
    def __init__(self, config=None):
        super().__init__(config or {"type": "my/style_transfer"})
        self.transform = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 4, 3, padding=1),
        )
    
    @classmethod
    def declare_io(cls):
        return {
            "x": InputPort("x"),
            "output": OutputPort("output"),
        }
    
    def process(self, **kw):
        return {"output": kw["x"] + self.transform(kw["x"])}

# 2. Insert into pipeline
graph = ComputeGraph.from_template("sd15_txt2img")
graph.add_node("style", StyleTransferBlock())
# Wire it between backbone and guidance
graph.connect("backbone", "output", "style", "x")
graph.connect("style", "output", "guidance", "model_output")

# 3. Train your block (everything else frozen!)
trainer = GraphTrainer(
    graph=graph,
    train_nodes=["style"],  # Only your block trains
    config=GraphTrainingConfig(learning_rate=1e-3, num_epochs=20),
)
trainer.train(dataset)
```

## Tutorial 6: FLUX.2 Generation

```python
from yggdrasil import Pipeline

# FLUX.2 text-to-image
pipe = Pipeline.from_template("flux2_txt2img", device="cuda", dtype=torch.bfloat16)
result = pipe(
    prompt="A photorealistic portrait of a cyberpunk samurai",
    num_steps=28,
    guidance_scale=3.5,
    width=1024,
    height=1024,
)
result.images[0].save("flux2_output.png")

# FLUX.2 Schnell (fast, no CFG needed)
pipe = Pipeline.from_template("flux2_schnell", device="cuda")
result = pipe(prompt="A cat wearing sunglasses", num_steps=4)
```

## Tutorial 7: Wan Video Generation

```python
from yggdrasil import Pipeline

# Text-to-video
pipe = Pipeline.from_template("wan_t2v", device="cuda")
result = pipe(
    prompt="A timelapse of a flower blooming",
    num_steps=50,
    num_frames=16,
)
# result.video contains the video tensor

# Image-to-video
pipe = Pipeline.from_template("wan_i2v", device="cuda")
result = pipe(
    prompt="A person walking",
    input_image=first_frame,
    num_frames=24,
)
```

## Tutorial 8: Launch Dynamic UI

```python
from yggdrasil.serving import DynamicUI

# Auto-generate UI from any template
ui = DynamicUI.from_template("sd15_txt2img")

# The UI automatically creates:
# - Input fields for all graph ports (prompt, guidance_scale, etc.)
# - Output gallery for generated images
# - Graph editor for modifying the pipeline
# - Training panel with per-node LR controls
# - Block browser showing all available Lego pieces

ui.launch(share=True)  # Creates a public URL
```

## Tutorial 9: Deploy to Cloud

```python
from yggdrasil import Pipeline

# Build your custom pipeline
graph = ComputeGraph.from_template("sd15_txt2img")
graph.replace_node("backbone", MyCustomBackbone())
apply_lora(graph, {"backbone": {"rank": 16}})

# Save the full pipeline
pipe = Pipeline.from_graph(graph)

# Deploy via REST API
from yggdrasil.serving import create_api
app = create_api()  # FastAPI app
# Run with: uvicorn app:app --host 0.0.0.0 --port 8000

# Or deploy via Gradio
from yggdrasil.serving import DynamicUI
ui = DynamicUI.from_pipeline(pipe)
ui.launch(server_name="0.0.0.0", server_port=7860, share=True)
```
