# Creating Custom Blocks

YggDrasil is a true Lego constructor — every component is a block that can be created, replaced, combined, and trained independently.

## Quick Start: Your First Custom Block

```python
import torch
import torch.nn as nn
from yggdrasil.core.block.base import AbstractBlock
from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import InputPort, OutputPort, Port


@register_block("my_blocks/custom_upscaler")
class CustomUpscaler(AbstractBlock):
    """A custom upscaling block."""
    
    block_type = "my_blocks/custom_upscaler"
    
    def __init__(self, config=None):
        config = config or {"type": "my_blocks/custom_upscaler", "scale_factor": 2}
        super().__init__(config)
        self.scale = int(self.config.get("scale_factor", 2))
        self.conv = nn.Conv2d(3, 3, 3, padding=1)
    
    @classmethod
    def declare_io(cls) -> dict:
        return {
            "image": InputPort("image", description="Input image tensor"),
            "output": OutputPort("output", description="Upscaled image"),
        }
    
    def process(self, **kw) -> dict:
        x = kw["image"]
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode="bilinear")
        x = self.conv(x)
        return {"output": x}
```

That's it! Your block is now:
- Registered in the global registry as `"my_blocks/custom_upscaler"`
- Usable in any ComputeGraph
- Trainable (the `conv` weights are automatically discovered)
- Serializable (config-based reconstruction)

## Block Contract

Every block must implement two methods:

### `declare_io()` → Dict[str, Port]
Declares the block's input and output ports. This is a **class method** that tells the system what data flows in and out.

```python
@classmethod
def declare_io(cls) -> dict:
    return {
        "x": InputPort("x", description="Input tensor"),
        "conditioning": InputPort("conditioning", optional=True),
        "output": OutputPort("output"),
    }
```

### `process(**port_inputs)` → dict
The actual computation. Receives named port values, returns a dict of outputs.

```python
def process(self, **kw) -> dict:
    x = kw["x"]
    cond = kw.get("conditioning")
    result = self.model(x, cond)
    return {"output": result}
```

## Using Your Block in a Graph

```python
from yggdrasil.core.graph.graph import ComputeGraph

# Load a standard pipeline
graph = ComputeGraph.from_template("sd15_txt2img")

# Replace the backbone with your custom one
graph.replace_node("backbone", MyCustomBackbone(config))

# Or add your block as a new node
graph.add_node("upscaler", CustomUpscaler({"scale_factor": 4}))
graph.connect("codec", "output", "upscaler", "image")

# Execute
outputs = graph.execute(prompt="hello world", num_steps=20)
```

## Custom Backbone

```python
@register_block("backbone/my_transformer")
class MyTransformerBackbone(AbstractBackbone):
    block_type = "backbone/my_transformer"
    
    def __init__(self, config):
        super().__init__(config)
        dim = int(config.get("hidden_dim", 768))
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, nhead=8, batch_first=True),
            num_layers=int(config.get("num_layers", 12)),
        )
    
    @classmethod
    def declare_io(cls):
        return {
            "x": InputPort("x", description="Noisy latent"),
            "t": InputPort("t", description="Timestep"),
            "context": InputPort("context", optional=True),
            "output": OutputPort("output", description="Predicted noise"),
        }
    
    def process(self, **kw):
        x = kw["x"]
        # Your transformer logic here
        return {"output": self.layers(x)}
```

## Custom Conditioner

```python
@register_block("conditioner/my_encoder")
class MyTextEncoder(AbstractConditioner):
    block_type = "conditioner/my_encoder"
    
    def __init__(self, config):
        super().__init__(config)
        # Your encoder initialization
    
    @classmethod
    def declare_io(cls):
        return {
            "text": InputPort("text", data_type="str"),
            "embeddings": OutputPort("embeddings"),
        }
    
    def process(self, **kw):
        text = kw["text"]
        # Encode text to embeddings
        return {"embeddings": encoded}
```

## Custom Guidance

```python
@register_block("guidance/my_guidance")
class MyGuidance(AbstractGuidance):
    block_type = "guidance/my_guidance"
    
    @classmethod
    def declare_io(cls):
        return {
            "model_output": InputPort("model_output"),
            "uncond_output": InputPort("uncond_output", optional=True),
            "guidance_scale": InputPort("guidance_scale", data_type="float", optional=True),
            "output": OutputPort("output"),
        }
    
    def process(self, **kw):
        cond = kw["model_output"]
        uncond = kw.get("uncond_output")
        scale = kw.get("guidance_scale", 7.5)
        
        if uncond is not None:
            guided = uncond + scale * (cond - uncond)
        else:
            guided = cond
        return {"output": guided}
```

## Training Your Custom Block

```python
from yggdrasil.training.graph_trainer import GraphTrainer, GraphTrainingConfig
from yggdrasil.training.data import ImageFolderSource

# Build graph with your custom block
graph = ComputeGraph.from_template("sd15_txt2img")
graph.replace_node("backbone", MyCustomBackbone(config))

# Train ONLY your block (everything else frozen)
trainer = GraphTrainer(
    graph=graph,
    train_nodes=["backbone"],
    config=GraphTrainingConfig(
        learning_rate=1e-4,
        num_epochs=100,
    ),
)
trainer.train(ImageFolderSource("/path/to/images"))
```

## Adding LoRA to Any Block

```python
from yggdrasil.blocks.adapters.lora import apply_lora, save_lora, load_lora

# Apply LoRA to your block
adapters = apply_lora(graph, {
    "backbone": {"rank": 16, "target_modules": ["to_q", "to_k", "to_v"]},
})

# Train only LoRA weights
trainer = GraphTrainer(graph, train_nodes=["backbone"])
trainer.train(dataset)

# Save/load LoRA separately
save_lora(graph, "my_lora/")
load_lora(graph, "my_lora/")
```
