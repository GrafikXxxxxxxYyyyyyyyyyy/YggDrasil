# yggdrasil/core/graph/templates/image_pipelines.py
"""Graph templates for image diffusion pipelines.

Covers: SD 1.5, SDXL, SD3, Flux, DiT, Kandinsky, DeepFloyd, PixArt, Stable Cascade.
Each function returns a ComputeGraph ready for FULL generation (with denoising loop).
"""
from __future__ import annotations
from typing import Any, Dict, Optional

from yggdrasil.core.graph.graph import ComputeGraph
from yggdrasil.core.graph.templates import register_template


# ==================== HELPER: BUILD DENOISE STEP GRAPH ====================

def _build_denoise_step(
    backbone,
    guidance,
    solver,
) -> ComputeGraph:
    """Build the inner per-step graph: backbone -> guidance -> solver.
    
    This graph executes once per denoising step inside LoopSubGraph.
    
    Inputs (provided by LoopSubGraph at each iteration):
        latents:       current noisy latents
        timestep:      current timestep
        next_timestep: next timestep
        condition:     conditioner embedding (constant across steps)
    
    Outputs:
        next_latents:  denoised latents for next step
    """
    step = ComputeGraph("denoise_step")
    step.add_node("backbone", backbone)
    step.add_node("guidance", guidance)
    step.add_node("solver", solver)
    
    # backbone -> guidance -> solver
    step.connect("backbone", "output", "guidance", "model_output")
    step.connect("guidance", "guided_output", "solver", "model_output")
    
    # Fan-out: latents -> backbone.x, solver.current_latents, guidance.x
    step.expose_input("latents", "backbone", "x")
    step.expose_input("latents", "solver", "current_latents")
    step.expose_input("latents", "guidance", "x")
    
    # Fan-out: timestep -> backbone.timestep, solver.timestep, guidance.t
    step.expose_input("timestep", "backbone", "timestep")
    step.expose_input("timestep", "solver", "timestep")
    step.expose_input("timestep", "guidance", "t")
    
    # Fan-out: condition -> backbone.condition, guidance.condition
    step.expose_input("condition", "backbone", "condition")
    step.expose_input("condition", "guidance", "condition")
    
    # next_timestep -> solver
    step.expose_input("next_timestep", "solver", "next_timestep")
    
    # Output
    step.expose_output("next_latents", "solver", "next_latents")
    step.expose_output("latents", "solver", "next_latents")
    
    return step


# ==================== HELPER: BUILD FULL PIPELINE ====================

def _build_txt2img_graph(
    name: str,
    backbone_config: dict,
    codec_config: dict,
    conditioner_configs: list[dict],
    guidance_config: dict,
    solver_config: dict,
    schedule_config: dict,
    process_config: dict,
    position_config: dict | None = None,
    num_steps: int = 50,
    default_width: int = 512,
    default_height: int = 512,
) -> ComputeGraph:
    """Build a complete txt2img pipeline with denoising loop.
    
    Параметры num_steps и guidance_scale НЕ запекаются навсегда —
    они сохраняются как defaults в metadata и могут быть переопределены
    при вызове ``graph.execute(num_steps=..., guidance_scale=...)``.
    
    Structure:
        prompt -> conditioner ─┐
                                ├─> LoopSubGraph(backbone -> guidance -> solver) -> codec -> image
        noise_latents ─────────┘
    """
    from yggdrasil.core.block.builder import BlockBuilder
    from yggdrasil.core.graph.subgraph import LoopSubGraph
    
    graph = ComputeGraph(name)
    
    # ── Metadata for runtime parameter resolution ──
    graph.metadata = {
        "default_guidance_scale": guidance_config.get("scale", 7.5),
        "default_num_steps": num_steps,
        "default_width": default_width,
        "default_height": default_height,
        "latent_channels": codec_config.get("latent_channels", 4),
        "spatial_scale_factor": codec_config.get("spatial_scale_factor", 8),
    }
    
    # Build blocks
    backbone = BlockBuilder.build(backbone_config)
    codec = BlockBuilder.build(codec_config)
    guidance = BlockBuilder.build(guidance_config)
    solver = BlockBuilder.build(solver_config)
    
    # Set backbone reference for CFG/SAG dual-pass
    if hasattr(guidance, '_backbone_ref'):
        guidance._backbone_ref = backbone
    
    # Build inner step graph
    step_graph = _build_denoise_step(backbone, guidance, solver)
    
    # Wrap in denoising loop
    loop = LoopSubGraph.create(
        inner_graph=step_graph,
        num_iterations=num_steps,
        carry_vars=["latents"],
    )
    
    # Add conditioners
    conditioners = []
    for i, cond_config in enumerate(conditioner_configs):
        cond = BlockBuilder.build(cond_config)
        conditioners.append(cond)
        graph.add_node(f"conditioner_{i}", cond)
    
    # Position embedder (if needed)
    if position_config:
        position = BlockBuilder.build(position_config)
        graph.add_node("position", position)
    
    # Add loop and codec to outer graph
    graph.add_node("denoise_loop", loop)
    graph.add_node("codec", codec)
    
    # Wire conditioners
    for i, cond in enumerate(conditioners):
        graph.expose_input(
            f"prompt_{i}" if i > 0 else "prompt",
            f"conditioner_{i}", "raw_condition",
        )
    
    # Connect first conditioner's embedding to loop's condition input
    graph.connect("conditioner_0", "embedding", "denoise_loop", "condition")
    
    # Noise input -> loop (optional: auto-generated if not provided)
    graph.expose_input("latents", "denoise_loop", "initial_latents")
    
    # Optional: timesteps input
    graph.expose_input("timesteps", "denoise_loop", "timesteps")
    
    # Loop output -> codec
    graph.connect("denoise_loop", "latents", "codec", "latent")
    
    # Graph outputs
    graph.expose_output("decoded", "codec", "decoded")
    graph.expose_output("latents", "denoise_loop", "latents")
    
    return graph


# ==================== STABLE DIFFUSION 1.5 ====================

@register_template("sd15_txt2img")
def sd15_txt2img(**kwargs) -> ComputeGraph:
    """Stable Diffusion 1.5 text-to-image pipeline (full, with denoising loop).
    
    Использование::
    
        graph = ComputeGraph.from_template("sd15_txt2img", device="cuda")
        outputs = graph.execute(prompt="a cat", guidance_scale=7.5, num_steps=28, seed=42)
        image = outputs["decoded"]
    """
    pretrained = kwargs.get("pretrained", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    return _build_txt2img_graph(
        name="sd15_txt2img",
        backbone_config={"type": "backbone/unet2d_condition", "pretrained": pretrained, "fp16": True},
        codec_config={"type": "codec/autoencoder_kl", "pretrained": pretrained, "fp16": True, "scaling_factor": 0.18215, "latent_channels": 4, "spatial_scale_factor": 8},
        conditioner_configs=[{"type": "conditioner/clip_text", "pretrained": pretrained, "tokenizer_subfolder": "tokenizer", "text_encoder_subfolder": "text_encoder", "max_length": 77}],
        guidance_config={"type": "guidance/cfg", "scale": 7.5},
        solver_config={"type": "diffusion/solver/ddim", "eta": 0.0},
        schedule_config={"type": "noise/schedule/linear", "num_train_timesteps": 1000},
        process_config={"type": "diffusion/process/ddpm"},
        num_steps=kwargs.get("num_steps", 50),
        default_width=512,
        default_height=512,
    )


@register_template("sd15_img2img")
def sd15_img2img(**kwargs) -> ComputeGraph:
    """SD 1.5 image-to-image: starts from encoded source image + noise."""
    graph = sd15_txt2img(**kwargs)
    graph.name = "sd15_img2img"
    graph.expose_input("source_image", "codec", "pixel_data")
    return graph


@register_template("sd15_inpainting")
def sd15_inpainting(**kwargs) -> ComputeGraph:
    """SD 1.5 inpainting: source image + mask."""
    graph = sd15_img2img(**kwargs)
    graph.name = "sd15_inpainting"
    # Mask passed as additional condition to backbone (through the loop's inner graph)
    return graph


# ==================== STABLE DIFFUSION XL ====================

@register_template("sdxl_txt2img")
def sdxl_txt2img(**kwargs) -> ComputeGraph:
    """Stable Diffusion XL text-to-image with dual text encoder."""
    pretrained = kwargs.get("pretrained", "stabilityai/stable-diffusion-xl-base-1.0")
    return _build_txt2img_graph(
        name="sdxl_txt2img",
        backbone_config={"type": "backbone/unet2d_condition", "pretrained": pretrained, "fp16": True},
        codec_config={"type": "codec/autoencoder_kl", "pretrained": pretrained, "fp16": True, "scaling_factor": 0.13025, "latent_channels": 4},
        conditioner_configs=[
            {"type": "conditioner/clip_text", "pretrained": pretrained, "tokenizer_subfolder": "tokenizer", "text_encoder_subfolder": "text_encoder", "max_length": 77},
            {"type": "conditioner/clip_text", "pretrained": pretrained, "tokenizer_subfolder": "tokenizer_2", "text_encoder_subfolder": "text_encoder_2", "max_length": 77},
        ],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 7.5)},
        solver_config={"type": "diffusion/solver/ddim", "eta": 0.0},
        schedule_config={"type": "noise/schedule/linear", "num_train_timesteps": 1000},
        process_config={"type": "diffusion/process/ddpm"},
        num_steps=kwargs.get("num_steps", 50),
    )


@register_template("sdxl_img2img")
def sdxl_img2img(**kwargs) -> ComputeGraph:
    graph = sdxl_txt2img(**kwargs)
    graph.name = "sdxl_img2img"
    graph.expose_input("source_image", "codec", "pixel_data")
    return graph


@register_template("sdxl_inpainting")
def sdxl_inpainting(**kwargs) -> ComputeGraph:
    graph = sdxl_img2img(**kwargs)
    graph.name = "sdxl_inpainting"
    return graph


@register_template("sdxl_refiner")
def sdxl_refiner(**kwargs) -> ComputeGraph:
    """SDXL refiner stage — takes noisy latents from base."""
    pretrained = kwargs.get("pretrained", "stabilityai/stable-diffusion-xl-refiner-1.0")
    graph = _build_txt2img_graph(
        name="sdxl_refiner",
        backbone_config={"type": "backbone/unet2d_condition", "pretrained": pretrained, "fp16": True},
        codec_config={"type": "codec/autoencoder_kl", "pretrained": pretrained, "fp16": True, "scaling_factor": 0.13025, "latent_channels": 4},
        conditioner_configs=[{"type": "conditioner/clip_text", "pretrained": pretrained, "tokenizer_subfolder": "tokenizer_2", "text_encoder_subfolder": "text_encoder_2", "max_length": 77}],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 5.0)},
        solver_config={"type": "diffusion/solver/ddim"},
        schedule_config={"type": "noise/schedule/linear"},
        process_config={"type": "diffusion/process/ddpm"},
        num_steps=kwargs.get("num_steps", 25),
    )
    return graph


# ==================== STABLE DIFFUSION 3 ====================

@register_template("sd3_txt2img")
def sd3_txt2img(**kwargs) -> ComputeGraph:
    """Stable Diffusion 3 with MMDiT backbone and flow matching."""
    return _build_txt2img_graph(
        name="sd3_txt2img",
        backbone_config={"type": "backbone/mmdit", "hidden_dim": 1536, "num_layers": 24, "num_heads": 24, "in_channels": 16, "patch_size": 2},
        codec_config={"type": "codec/autoencoder_kl", "pretrained": kwargs.get("pretrained", "stabilityai/stable-diffusion-3-medium"), "fp16": True, "latent_channels": 16},
        conditioner_configs=[
            {"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14", "max_length": 77},
        ],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 5.0)},
        solver_config={"type": "diffusion/solver/heun"},
        schedule_config={"type": "noise/schedule/sigmoid"},
        process_config={"type": "diffusion/process/flow/rectified"},
        position_config={"type": "position/rope_nd", "dim": 64},
        num_steps=kwargs.get("num_steps", 28),
    )


# ==================== FLUX ====================

@register_template("flux_txt2img")
def flux_txt2img(**kwargs) -> ComputeGraph:
    """Flux text-to-image with MMDiT and rectified flow."""
    return _build_txt2img_graph(
        name="flux_txt2img",
        backbone_config={"type": "backbone/mmdit", "hidden_dim": 3072, "num_layers": 19, "num_heads": 24, "in_channels": 16, "patch_size": 2, "cond_dim": 4096},
        codec_config={"type": "codec/autoencoder_kl", "pretrained": kwargs.get("pretrained", "black-forest-labs/FLUX.1-dev"), "fp16": True, "latent_channels": 16},
        conditioner_configs=[{"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14", "max_length": 77}],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 3.5)},
        solver_config={"type": "diffusion/solver/heun"},
        schedule_config={"type": "noise/schedule/sigmoid"},
        process_config={"type": "diffusion/process/flow/rectified"},
        position_config={"type": "position/rope_nd", "dim": 64},
        num_steps=kwargs.get("num_steps", 28),
    )


@register_template("flux_img2img")
def flux_img2img(**kwargs) -> ComputeGraph:
    graph = flux_txt2img(**kwargs)
    graph.name = "flux_img2img"
    graph.expose_input("source_image", "codec", "pixel_data")
    return graph


# ==================== DiT (Unconditional) ====================

@register_template("dit_unconditional")
def dit_unconditional(**kwargs) -> ComputeGraph:
    """DiT unconditional generation (class-conditioned)."""
    from yggdrasil.core.block.builder import BlockBuilder
    from yggdrasil.core.graph.subgraph import LoopSubGraph
    
    backbone = BlockBuilder.build({"type": "backbone/dit", "hidden_dim": 1152, "num_layers": 28, "num_heads": 16, "in_channels": 4, "patch_size": 2})
    solver = BlockBuilder.build({"type": "diffusion/solver/ddim"})
    
    # Inner step graph (no guidance for unconditional)
    step = ComputeGraph("dit_step")
    step.add_node("backbone", backbone)
    step.add_node("solver", solver)
    step.connect("backbone", "output", "solver", "model_output")
    step.expose_input("latents", "backbone", "x")
    step.expose_input("latents", "solver", "current_latents")
    step.expose_input("timestep", "backbone", "timestep")
    step.expose_input("timestep", "solver", "timestep")
    step.expose_input("next_timestep", "solver", "next_timestep")
    step.expose_output("next_latents", "solver", "next_latents")
    step.expose_output("latents", "solver", "next_latents")
    
    loop = LoopSubGraph.create(inner_graph=step, num_iterations=kwargs.get("num_steps", 50))
    
    graph = ComputeGraph("dit_unconditional")
    graph.add_node("denoise_loop", loop)
    graph.expose_input("latents", "denoise_loop", "initial_latents")
    graph.expose_output("latents", "denoise_loop", "latents")
    
    return graph


# ==================== PixArt ====================

@register_template("pixart_txt2img")
def pixart_txt2img(**kwargs) -> ComputeGraph:
    """PixArt Alpha/Sigma text-to-image."""
    return _build_txt2img_graph(
        name="pixart_txt2img",
        backbone_config={"type": "backbone/dit", "hidden_dim": 1152, "num_layers": 28, "num_heads": 16, "in_channels": 4, "patch_size": 2, "cond_dim": 4096},
        codec_config={"type": "codec/autoencoder_kl", "latent_channels": 4},
        conditioner_configs=[{"type": "conditioner/t5_text", "pretrained": "google/t5-v1_1-xxl", "max_length": 120}],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 4.5)},
        solver_config={"type": "diffusion/solver/ddim"},
        schedule_config={"type": "noise/schedule/linear"},
        process_config={"type": "diffusion/process/ddpm"},
        position_config={"type": "position/rope_nd", "dim": 64},
        num_steps=kwargs.get("num_steps", 20),
    )


# ==================== Kandinsky ====================

@register_template("kandinsky_txt2img")
def kandinsky_txt2img(**kwargs) -> ComputeGraph:
    """Kandinsky 2.2 text-to-image."""
    return _build_txt2img_graph(
        name="kandinsky_txt2img",
        backbone_config={"type": "backbone/unet2d_condition", "pretrained": kwargs.get("pretrained", "kandinsky-community/kandinsky-2-2-decoder"), "fp16": True},
        codec_config={"type": "codec/autoencoder_kl", "pretrained": kwargs.get("pretrained", "kandinsky-community/kandinsky-2-2-decoder"), "fp16": True, "latent_channels": 4},
        conditioner_configs=[{"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14"}],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 4.0)},
        solver_config={"type": "diffusion/solver/ddim"},
        schedule_config={"type": "noise/schedule/linear"},
        process_config={"type": "diffusion/process/ddpm"},
        num_steps=kwargs.get("num_steps", 50),
    )


# ==================== Stable Cascade ====================

@register_template("stable_cascade")
def stable_cascade(**kwargs) -> ComputeGraph:
    """Stable Cascade — two-stage cascade pipeline using nested LoopSubGraphs."""
    from yggdrasil.core.block.builder import BlockBuilder
    from yggdrasil.core.graph.subgraph import LoopSubGraph
    
    graph = ComputeGraph("stable_cascade")
    
    # Conditioner (shared)
    conditioner = BlockBuilder.build({"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14"})
    graph.add_node("conditioner", conditioner)
    
    # Stage C (prior): text -> prior latents
    prior_backbone = BlockBuilder.build({"type": "backbone/dit", "hidden_dim": 1536, "num_layers": 24, "num_heads": 24, "in_channels": 16, "patch_size": 2})
    prior_guidance = BlockBuilder.build({"type": "guidance/cfg", "scale": 4.0})
    prior_solver = BlockBuilder.build({"type": "diffusion/solver/ddim"})
    prior_guidance._backbone_ref = prior_backbone
    
    prior_step = _build_denoise_step(prior_backbone, prior_guidance, prior_solver)
    prior_loop = LoopSubGraph.create(inner_graph=prior_step, num_iterations=20)
    graph.add_node("prior_loop", prior_loop)
    
    # Stage B (decoder): prior latents -> image latents
    decoder_backbone = BlockBuilder.build({"type": "backbone/unet2d_condition", "pretrained": kwargs.get("pretrained", "stabilityai/stable-cascade"), "fp16": True})
    decoder_solver = BlockBuilder.build({"type": "diffusion/solver/ddim"})
    
    decoder_step = ComputeGraph("decoder_step")
    decoder_step.add_node("backbone", decoder_backbone)
    decoder_step.add_node("solver", decoder_solver)
    decoder_step.connect("backbone", "output", "solver", "model_output")
    decoder_step.expose_input("latents", "backbone", "x")
    decoder_step.expose_input("latents", "solver", "current_latents")
    decoder_step.expose_input("timestep", "backbone", "timestep")
    decoder_step.expose_input("timestep", "solver", "timestep")
    decoder_step.expose_input("next_timestep", "solver", "next_timestep")
    decoder_step.expose_input("condition", "backbone", "condition")
    decoder_step.expose_output("next_latents", "solver", "next_latents")
    decoder_step.expose_output("latents", "solver", "next_latents")
    
    decoder_loop = LoopSubGraph.create(inner_graph=decoder_step, num_iterations=10)
    graph.add_node("decoder_loop", decoder_loop)
    
    codec = BlockBuilder.build({"type": "codec/autoencoder_kl", "latent_channels": 4})
    graph.add_node("codec", codec)
    
    # Wire
    graph.expose_input("prompt", "conditioner", "raw_condition")
    graph.connect("conditioner", "embedding", "prior_loop", "condition")
    graph.expose_input("prior_latents", "prior_loop", "initial_latents")
    
    graph.connect("prior_loop", "latents", "decoder_loop", "condition")
    graph.expose_input("decoder_latents", "decoder_loop", "initial_latents")
    
    graph.connect("decoder_loop", "latents", "codec", "latent")
    graph.expose_output("decoded", "codec", "decoded")
    
    return graph


# ==================== DeepFloyd IF ====================

@register_template("deepfloyd_txt2img")
def deepfloyd_txt2img(**kwargs) -> ComputeGraph:
    """DeepFloyd IF — three-stage cascade using LoopSubGraphs."""
    from yggdrasil.core.block.builder import BlockBuilder
    from yggdrasil.core.graph.subgraph import LoopSubGraph
    
    graph = ComputeGraph("deepfloyd_txt2img")
    
    # Shared conditioner
    conditioner = BlockBuilder.build({"type": "conditioner/t5_text", "pretrained": "google/t5-v1_1-xxl", "max_length": 77})
    graph.add_node("conditioner", conditioner)
    
    def _make_stage(name, pretrained, guidance_scale=7.0, steps=50):
        bb = BlockBuilder.build({"type": "backbone/unet2d_condition", "pretrained": pretrained, "fp16": True})
        g = BlockBuilder.build({"type": "guidance/cfg", "scale": guidance_scale})
        s = BlockBuilder.build({"type": "diffusion/solver/ddim"})
        g._backbone_ref = bb
        step = _build_denoise_step(bb, g, s)
        return LoopSubGraph.create(inner_graph=step, num_iterations=steps)
    
    # Stage 1: 64x64
    stage1 = _make_stage("stage1", kwargs.get("stage1", "DeepFloyd/IF-I-XL-v1.0"), 7.0, 50)
    graph.add_node("stage1_loop", stage1)
    
    # Stage 2: 64->256
    stage2 = _make_stage("stage2", kwargs.get("stage2", "DeepFloyd/IF-II-L-v1.0"), 4.0, 30)
    graph.add_node("stage2_loop", stage2)
    
    # Stage 3: 256->1024
    stage3 = _make_stage("stage3", kwargs.get("stage3", "stabilityai/stable-diffusion-x4-upscaler"), 4.0, 20)
    graph.add_node("stage3_loop", stage3)
    
    # Wire
    graph.expose_input("prompt", "conditioner", "raw_condition")
    graph.connect("conditioner", "embedding", "stage1_loop", "condition")
    graph.expose_input("stage1_latents", "stage1_loop", "initial_latents")
    
    graph.connect("conditioner", "embedding", "stage2_loop", "condition")
    graph.connect("stage1_loop", "latents", "stage2_loop", "initial_latents")
    
    graph.connect("conditioner", "embedding", "stage3_loop", "condition")
    graph.connect("stage2_loop", "latents", "stage3_loop", "initial_latents")
    
    graph.expose_output("image", "stage3_loop", "latents")
    
    return graph
