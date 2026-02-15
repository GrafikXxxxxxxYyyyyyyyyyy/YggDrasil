# yggdrasil/core/graph/templates/training_pipelines.py
"""Training recipe templates — complete training graphs as ComputeGraphs.

Each template builds a complete training graph that includes:
- DatasetBlock (data source)
- Forward pass (backbone, conditioner, etc.)
- LossBlock (loss computation)

These can be trained with GraphTrainer:

    graph = ComputeGraph.from_template("train_lora_sd15")
    trainer = GraphTrainer(graph, train_nodes=["lora_adapter"])
    trainer.train(dataset)
"""
from __future__ import annotations

from typing import Any
from ..graph import ComputeGraph
from . import register_template


@register_template("train_lora_sd15")
def build_train_lora_sd15(**kwargs) -> ComputeGraph:
    """LoRA training graph for Stable Diffusion 1.5.
    
    Trains a LoRA adapter on the backbone while keeping everything else frozen.
    
    Graph structure:
        dataset -> conditioner -> backbone (+ LoRA) -> loss
    """
    from yggdrasil.core.block.builder import BlockBuilder
    
    graph = ComputeGraph("train_lora_sd15")
    graph.metadata.update({
        "task": "training",
        "method": "lora",
        "base_model": "sd15",
        "default_train_nodes": ["lora_adapter"],
        "default_lr": 1e-4,
        "default_epochs": 100,
    })
    
    # Conditioner (frozen)
    graph.add_node("conditioner", BlockBuilder.build({
        "type": "conditioner/clip_text",
        "pretrained": kwargs.get("pretrained_text_encoder", "openai/clip-vit-large-patch14"),
    }))
    
    # Backbone (frozen, but LoRA is applied)
    graph.add_node("backbone", BlockBuilder.build({
        "type": "backbone/unet2d_condition",
        "pretrained": kwargs.get("pretrained_backbone", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
    }))
    
    # Loss
    graph.add_node("loss", BlockBuilder.build({"type": "loss/mse"}))
    
    # Edges
    graph.connect("conditioner", "embedding", "backbone", "condition")
    graph.connect("backbone", "output", "loss", "prediction")
    
    # Inputs
    graph.expose_input("prompt", "conditioner", "text")
    graph.expose_input("latents", "backbone", "x")
    graph.expose_input("timestep", "backbone", "timestep")
    graph.expose_input("target", "loss", "target")
    
    # Outputs
    graph.expose_output("loss", "loss", "loss")
    graph.expose_output("prediction", "backbone", "output")
    
    return graph


@register_template("train_lora_sdxl")
def build_train_lora_sdxl(**kwargs) -> ComputeGraph:
    """LoRA training graph for Stable Diffusion XL."""
    from yggdrasil.core.block.builder import BlockBuilder

    graph = ComputeGraph("train_lora_sdxl")
    graph.metadata.update({
        "task": "training",
        "method": "lora",
        "base_model": "sdxl",
        "default_train_nodes": ["lora_adapter"],
        "default_lr": 1e-4,
        "default_epochs": 100,
    })

    pretrained = kwargs.get("pretrained_backbone", "stabilityai/stable-diffusion-xl-base-1.0")
    graph.add_node("conditioner", BlockBuilder.build({
        "type": "conditioner/clip_sdxl",
        "pretrained": pretrained,
    }))
    graph.add_node("backbone", BlockBuilder.build({
        "type": "backbone/unet2d_condition",
        "pretrained": pretrained,
    }))
    graph.add_node("loss", BlockBuilder.build({"type": "loss/mse"}))

    graph.connect("conditioner", "condition", "backbone", "condition")
    graph.connect("backbone", "output", "loss", "prediction")
    graph.expose_input("prompt", "conditioner", "prompt")
    graph.expose_input("latents", "backbone", "x")
    graph.expose_input("timestep", "backbone", "timestep")
    graph.expose_input("target", "loss", "target")
    graph.expose_output("loss", "loss", "loss")
    graph.expose_output("prediction", "backbone", "output")
    return graph


@register_template("train_lora_flux")
def build_train_lora_flux(**kwargs) -> ComputeGraph:
    """LoRA training graph for FLUX.1 (transformer backbone)."""
    from yggdrasil.core.block.builder import BlockBuilder

    graph = ComputeGraph("train_lora_flux")
    graph.metadata.update({
        "task": "training",
        "method": "lora",
        "base_model": "flux",
        "default_train_nodes": ["backbone"],
        "default_lr": 1e-4,
        "default_epochs": 100,
    })

    pretrained = kwargs.get("pretrained_backbone", "black-forest-labs/FLUX.1-dev")
    graph.add_node("conditioner", BlockBuilder.build({
        "type": "conditioner/clip_text",
        "pretrained": "openai/clip-vit-large-patch14",
        "max_length": 77,
    }))
    graph.add_node("backbone", BlockBuilder.build({
        "type": "backbone/flux_transformer",
        "pretrained": pretrained,
        "fp16": kwargs.get("fp16", True),
    }))
    graph.add_node("loss", BlockBuilder.build({"type": "loss/mse"}))

    graph.connect("conditioner", "embedding", "backbone", "condition")
    graph.connect("backbone", "output", "loss", "prediction")
    graph.expose_input("prompt", "conditioner", "raw_condition")
    graph.expose_input("latents", "backbone", "x")
    graph.expose_input("timestep", "backbone", "timestep")
    graph.expose_input("target", "loss", "target")
    graph.expose_output("loss", "loss", "loss")
    graph.expose_output("prediction", "backbone", "output")
    return graph


@register_template("train_lora_sd3")
def build_train_lora_sd3(**kwargs) -> ComputeGraph:
    """LoRA training graph for Stable Diffusion 3 (MMDiT backbone)."""
    from yggdrasil.core.block.builder import BlockBuilder

    graph = ComputeGraph("train_lora_sd3")
    graph.metadata.update({
        "task": "training",
        "method": "lora",
        "base_model": "sd3",
        "default_train_nodes": ["backbone"],
        "default_lr": 1e-4,
        "default_epochs": 100,
    })

    graph.add_node("conditioner", BlockBuilder.build({
        "type": "conditioner/clip_text",
        "pretrained": "openai/clip-vit-large-patch14",
        "max_length": 77,
    }))
    graph.add_node("backbone", BlockBuilder.build({
        "type": "backbone/mmdit",
        "hidden_dim": 1536,
        "num_layers": 24,
        "num_heads": 24,
        "in_channels": 16,
        "patch_size": 2,
    }))
    graph.add_node("loss", BlockBuilder.build({"type": "loss/mse"}))

    graph.connect("conditioner", "embedding", "backbone", "condition")
    graph.connect("backbone", "output", "loss", "prediction")
    graph.expose_input("prompt", "conditioner", "raw_condition")
    graph.expose_input("latents", "backbone", "x")
    graph.expose_input("timestep", "backbone", "timestep")
    graph.expose_input("target", "loss", "target")
    graph.expose_output("loss", "loss", "loss")
    graph.expose_output("prediction", "backbone", "output")
    return graph


@register_template("train_textual_inversion")
def build_train_textual_inversion(**kwargs) -> ComputeGraph:
    """Textual Inversion training graph.
    
    Trains a new text embedding for a concept while backbone is frozen.
    
    Graph structure:
        dataset -> conditioner (with TI embed) -> backbone -> loss
    """
    from yggdrasil.core.block.builder import BlockBuilder
    
    graph = ComputeGraph("train_textual_inversion")
    graph.metadata.update({
        "task": "training",
        "method": "textual_inversion",
        "base_model": "sd15",
        "default_train_nodes": ["ti_adapter"],
        "default_lr": 5e-4,
        "default_epochs": 3000,
    })
    
    # Conditioner (frozen)
    graph.add_node("conditioner", BlockBuilder.build({
        "type": "conditioner/clip_text",
        "pretrained": kwargs.get("pretrained_text_encoder", "openai/clip-vit-large-patch14"),
    }))
    
    # TI Adapter (trainable)
    graph.add_node("ti_adapter", BlockBuilder.build({
        "type": "adapter/textual_inversion",
        "placeholder_token": kwargs.get("placeholder_token", "<concept>"),
        "num_vectors": kwargs.get("num_vectors", 1),
        "embedding_dim": kwargs.get("embedding_dim", 768),
    }))
    
    # Backbone (frozen)
    graph.add_node("backbone", BlockBuilder.build({
        "type": "backbone/unet2d_condition",
        "pretrained": kwargs.get("pretrained_backbone", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
    }))
    
    # Loss
    graph.add_node("loss", BlockBuilder.build({"type": "loss/mse"}))
    
    # Edges
    graph.connect("conditioner", "embedding", "ti_adapter", "text_embeds")
    graph.connect("ti_adapter", "modified_embeds", "backbone", "condition")
    graph.connect("backbone", "output", "loss", "prediction")
    
    # Inputs
    graph.expose_input("prompt", "conditioner", "text")
    graph.expose_input("latents", "backbone", "x")
    graph.expose_input("timestep", "backbone", "timestep")
    graph.expose_input("target", "loss", "target")
    
    # Outputs
    graph.expose_output("loss", "loss", "loss")
    
    return graph


@register_template("train_controlnet")
def build_train_controlnet(**kwargs) -> ComputeGraph:
    """ControlNet training graph.
    
    Trains a ControlNet adapter using paired (condition_image, target) data.
    Backbone is frozen, only ControlNet weights are trainable.
    
    Graph structure:
        dataset -> conditioner -> backbone (frozen) + controlnet (trainable) -> loss
    """
    from yggdrasil.core.block.builder import BlockBuilder
    
    graph = ComputeGraph("train_controlnet")
    graph.metadata.update({
        "task": "training",
        "method": "controlnet",
        "base_model": "sd15",
        "default_train_nodes": ["controlnet"],
        "default_lr": 1e-5,
        "default_epochs": 100,
    })
    
    # Conditioner
    graph.add_node("conditioner", BlockBuilder.build({
        "type": "conditioner/clip_text",
        "pretrained": kwargs.get("pretrained_text_encoder", "openai/clip-vit-large-patch14"),
    }))
    
    # ControlNet (trainable)
    graph.add_node("controlnet", BlockBuilder.build({
        "type": "adapter/controlnet",
        "control_type": kwargs.get("control_type", "depth"),
    }))
    
    # Backbone (frozen)
    graph.add_node("backbone", BlockBuilder.build({
        "type": "backbone/unet2d_condition",
        "pretrained": kwargs.get("pretrained_backbone", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
    }))
    
    # Loss
    graph.add_node("loss", BlockBuilder.build({"type": "loss/mse"}))
    
    # Edges
    graph.connect("conditioner", "embedding", "backbone", "condition")
    graph.connect("controlnet", "down_block_residuals", "backbone", "down_block_residuals")
    graph.connect("controlnet", "mid_block_residual", "backbone", "mid_block_residual")
    graph.connect("backbone", "output", "loss", "prediction")
    
    # Inputs
    graph.expose_input("prompt", "conditioner", "text")
    graph.expose_input("control_image", "controlnet", "control_image")
    graph.expose_input("latents", "backbone", "x")
    graph.expose_input("timestep", "backbone", "timestep")
    graph.expose_input("target", "loss", "target")
    
    # Outputs
    graph.expose_output("loss", "loss", "loss")
    
    return graph


@register_template("train_full_finetune")
def build_train_full_finetune(**kwargs) -> ComputeGraph:
    """Full model fine-tuning graph.
    
    Trains the full backbone (all parameters). Memory intensive but highest quality.
    
    Graph structure:
        dataset -> conditioner -> backbone (trainable) -> loss
    """
    from yggdrasil.core.block.builder import BlockBuilder
    
    graph = ComputeGraph("train_full_finetune")
    graph.metadata.update({
        "task": "training",
        "method": "full_finetune",
        "base_model": kwargs.get("base_model", "sd15"),
        "default_train_nodes": ["backbone"],
        "default_lr": 1e-6,
        "default_epochs": 10,
    })
    
    # Conditioner (frozen by default)
    graph.add_node("conditioner", BlockBuilder.build({
        "type": "conditioner/clip_text",
        "pretrained": kwargs.get("pretrained_text_encoder", "openai/clip-vit-large-patch14"),
    }))
    
    # Backbone (trainable)
    graph.add_node("backbone", BlockBuilder.build({
        "type": "backbone/unet2d_condition",
        "pretrained": kwargs.get("pretrained_backbone", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
    }))
    
    # Loss
    graph.add_node("loss", BlockBuilder.build({"type": "loss/mse"}))
    
    # Edges
    graph.connect("conditioner", "embedding", "backbone", "condition")
    graph.connect("backbone", "output", "loss", "prediction")
    
    # Inputs
    graph.expose_input("prompt", "conditioner", "text")
    graph.expose_input("latents", "backbone", "x")
    graph.expose_input("timestep", "backbone", "timestep")
    graph.expose_input("target", "loss", "target")
    
    # Outputs
    graph.expose_output("loss", "loss", "loss")
    graph.expose_output("prediction", "backbone", "output")
    
    return graph


@register_template("train_hypernetwork")
def build_train_hypernetwork(**kwargs) -> ComputeGraph:
    """Hypernetwork training graph.
    
    Trains a hypernetwork adapter that modifies cross-attention.
    """
    from yggdrasil.core.block.builder import BlockBuilder
    
    graph = ComputeGraph("train_hypernetwork")
    graph.metadata.update({
        "task": "training",
        "method": "hypernetwork",
        "base_model": "sd15",
        "default_train_nodes": ["hypernetwork"],
        "default_lr": 5e-5,
        "default_epochs": 500,
    })
    
    # Conditioner (frozen)
    graph.add_node("conditioner", BlockBuilder.build({
        "type": "conditioner/clip_text",
        "pretrained": kwargs.get("pretrained_text_encoder", "openai/clip-vit-large-patch14"),
    }))
    
    # Hypernetwork (trainable)
    graph.add_node("hypernetwork", BlockBuilder.build({
        "type": "adapter/hypernetwork",
        "hidden_size": kwargs.get("hidden_size", 768),
        "layer_structure": kwargs.get("layer_structure", [768, 128, 768]),
        "activation": kwargs.get("activation", "relu"),
    }))
    
    # Backbone (frozen)
    graph.add_node("backbone", BlockBuilder.build({
        "type": "backbone/unet2d_condition",
        "pretrained": kwargs.get("pretrained_backbone", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
    }))
    
    # Loss
    graph.add_node("loss", BlockBuilder.build({"type": "loss/mse"}))
    
    # Edges
    graph.connect("conditioner", "embedding", "backbone", "condition")
    graph.connect("backbone", "output", "loss", "prediction")
    
    # Inputs
    graph.expose_input("prompt", "conditioner", "text")
    graph.expose_input("latents", "backbone", "x")
    graph.expose_input("timestep", "backbone", "timestep")
    graph.expose_input("target", "loss", "target")
    
    # Outputs
    graph.expose_output("loss", "loss", "loss")
    
    return graph
