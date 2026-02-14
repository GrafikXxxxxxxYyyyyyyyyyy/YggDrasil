# API Reference

## Pipeline — High-Level API (Diffusers-like)

The simplest way to use YggDrasil. One class for any diffusion task.

```python
from yggdrasil import Pipeline

# From template
pipe = Pipeline.from_template("sd15_txt2img", device="cuda", dtype=torch.float16)
result = pipe(prompt="a cat in space", num_steps=28, guidance_scale=7.5)
result.images  # List[PIL.Image]

# From pretrained (HuggingFace)
pipe = Pipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
result = pipe("beautiful sunset over mountains")

# From custom graph
graph = ComputeGraph.from_template("sd15_txt2img")
graph.replace_node("backbone", MyBackbone())
pipe = Pipeline.from_graph(graph, device="cuda")
result = pipe(prompt="test")
```

### Pipeline Methods

| Method | Description |
|--------|-------------|
| `Pipeline.from_template(name, device, dtype)` | Create from registered template |
| `Pipeline.from_pretrained(model_id)` | Load from HuggingFace Hub |
| `Pipeline.from_graph(graph, device, dtype)` | Wrap existing ComputeGraph |
| `pipe(prompt, ...)` | Generate (alias for `__call__`) |
| `pipe.to(device, dtype)` | Move to device |

### Pipeline.__call__ Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Text prompt |
| `negative_prompt` | str | "" | Negative prompt |
| `num_steps` | int | 28 | Inference steps |
| `guidance_scale` | float | 7.5 | CFG scale |
| `width` | int | 512 | Output width |
| `height` | int | 512 | Output height |
| `seed` | int | None | Random seed |
| `latents` | Tensor | None | Pre-generated latents |

---

## ComputeGraph — The Core

The ComputeGraph is the fundamental data structure — a DAG of blocks connected by ports.

```python
from yggdrasil.core.graph.graph import ComputeGraph

# Create from template
graph = ComputeGraph.from_template("sd15_txt2img")

# Manipulate
graph.add_node("my_block", MyBlock(config))
graph.replace_node("backbone", MyBackbone())
graph.remove_node("old_block")
graph.connect("source_node", "output_port", "dest_node", "input_port")

# Move to device
graph.to("cuda", torch.float16)

# Execute (high-level, delegates to Pipeline)
outputs = graph.execute(prompt="test", guidance_scale=7.5)

# Execute (low-level, direct)
outputs = graph.execute_raw(x=latents, t=timestep, context=embeddings)

# Visualize
print(graph.visualize())  # Mermaid diagram
```

---

## GraphTrainer — Train Any Component

```python
from yggdrasil.training import GraphTrainer, GraphTrainingConfig

trainer = GraphTrainer(
    graph=graph,
    train_nodes=["backbone", "adapter"],
    config=GraphTrainingConfig(
        num_epochs=100,
        learning_rate=1e-4,
        node_lr={"backbone": 1e-5, "adapter": 1e-4},  # Per-node LR
        schedule=[
            {"epoch": 0, "freeze": ["backbone"]},       # Phase 1: adapter only
            {"epoch": 50, "unfreeze": ["backbone"]},     # Phase 2: joint
            {"epoch": 50, "set_lr": {"backbone": 1e-6}}, # Lower LR for backbone
        ],
        optimizer="adamw",
        mixed_precision="fp16",
        gradient_accumulation_steps=4,
        use_ema=True,
    ),
)

history = trainer.train(dataset)
trainer.save_checkpoint("my_training")
trainer.save_trained_nodes("exported_weights/")
```

---

## LoRA — Universal Adapter

```python
from yggdrasil.blocks.adapters.lora import (
    LoRAAdapter, apply_lora, save_lora, load_lora, 
    merge_lora, unmerge_lora, get_lora_info
)

# Apply to single block
lora = LoRAAdapter({"rank": 16, "target_modules": ["to_q", "to_k", "to_v"]})
lora.inject_into(graph.nodes["backbone"])

# Apply to multiple blocks at once
adapters = apply_lora(graph, {
    "backbone": {"rank": 16, "target_modules": ["to_q", "to_k", "to_v"]},
    "text_encoder": {"rank": 8, "target_modules": ["q_proj", "v_proj"]},
})

# Save/Load
save_lora(graph, "my_lora/")
adapters = load_lora(graph, "my_lora/")

# Merge for deployment (no LoRA overhead)
merge_lora(graph)

# Info
info = get_lora_info(graph)
# {"backbone": {"rank": 16, "num_layers": 24, "num_parameters": 1234567}}
```

---

## Dynamic Gradio UI

```python
from yggdrasil.serving import DynamicUI

# Auto-generate UI from graph
ui = DynamicUI.from_template("sd15_txt2img")
ui.launch(share=True)

# From pipeline
pipe = Pipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
ui = DynamicUI.from_pipeline(pipe)
ui.launch()

# Add custom tabs
def my_tab(gr):
    gr.Markdown("### My Custom Tab")
    gr.Textbox(label="Custom Input")

ui = DynamicUI(graph=graph)
ui.add_tab("My Tab", my_tab)
ui.launch()
```

---

## Available Templates

| Template | Model | Description |
|----------|-------|-------------|
| `sd15_txt2img` | SD 1.5 | Text-to-image |
| `sd15_img2img` | SD 1.5 | Image-to-image |
| `sd15_inpaint` | SD 1.5 | Inpainting |
| `sdxl_txt2img` | SDXL | Text-to-image |
| `sd3_txt2img` | SD 3 | DiT text-to-image |
| `flux2_txt2img` | FLUX.2 | MMDiT text-to-image |
| `flux2_schnell` | FLUX.2 Schnell | Fast (no CFG) |
| `flux2_fill` | FLUX.2 | Inpainting |
| `flux2_canny` | FLUX.2 | Canny ControlNet |
| `flux2_depth` | FLUX.2 | Depth ControlNet |
| `flux2_redux` | FLUX.2 | Image variation |
| `flux2_kontext` | FLUX.2 | Context editing |
| `wan_t2v` | Wan | Text-to-video |
| `wan_i2v` | Wan | Image-to-video |
| `wan_flf2v` | Wan | First-last-frame |
| `wan_fun_control` | Wan | ControlNet video |
| `qwen_image_txt2img` | QwenImage | Text-to-image |
| `qwen_image_edit` | QwenImage | Image editing |
| `controlnet_txt2img` | SD 1.5 | ControlNet |
| `stable_cascade` | Cascade | Two-stage |
| `deepfloyd` | DeepFloyd | Super-resolution |

---

## Distributed Training

```python
from yggdrasil.training.distributed import (
    setup_distributed, wrap_ddp, wrap_fsdp, DeepSpeedGraphTrainer
)

# DDP
setup_distributed()
graph = wrap_ddp(graph, train_nodes=["backbone"])

# FSDP (for very large models)
graph = wrap_fsdp(graph, train_nodes=["backbone"], 
                  sharding_strategy="FULL_SHARD", cpu_offload=True)

# DeepSpeed
trainer = DeepSpeedGraphTrainer(graph, train_nodes=["backbone"], ds_config={
    "train_batch_size": 4,
    "zero_optimization": {"stage": 2},
    "fp16": {"enabled": True},
})
```

---

## Block Registry

All blocks are registered with `@register_block("category/name")` and can be discovered:

```python
from yggdrasil.core.block.registry import BlockRegistry, list_blocks

# List all registered blocks
for name, cls in sorted(list_blocks().items()):
    print(f"{name}: {cls.__doc__}")

# Build any block from config
from yggdrasil.core.block.builder import BlockBuilder
block = BlockBuilder.build({"type": "backbone/unet2d", "in_channels": 4})
```
