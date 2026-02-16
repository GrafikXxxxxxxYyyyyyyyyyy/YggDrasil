# yggdrasil/core/graph/templates/image_pipelines.py
"""Graph templates for image diffusion pipelines.

Covers: SD 1.5, SDXL, SD3, Flux, DiT, Kandinsky, DeepFloyd, PixArt, Stable Cascade.
Each function returns a ComputeGraph ready for FULL generation (with denoising loop).
"""
from __future__ import annotations
from typing import Any, Dict, Optional

from yggdrasil.core.graph.graph import ComputeGraph
from yggdrasil.core.graph.templates import register_template
from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.base import AbstractBaseBlock
from yggdrasil.core.block.port import Port


def _require_diffusers_klein():
    """Проверка, что установлена версия diffusers с поддержкой FLUX.2 [klein] (AutoencoderKLFlux2, Flux2Transformer2DModel)."""
    try:
        import diffusers
        from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
    except ImportError as e:
        raise ImportError(
            "FLUX.2 [klein] требует diffusers из ветки main (0.37.0.dev0+). "
            "Установите: pip install 'git+https://github.com/huggingface/diffusers.git' "
            "или: pip install -e '.[klein]'"
        ) from e
    ver = getattr(diffusers, "__version__", "")
    # 0.36.0 с PyPI не содержит AutoencoderKLFlux2; нужна dev-версия
    if ver.startswith("0.36.") and "dev" not in ver:
        raise ImportError(
            f"Для FLUX.2 [klein] нужна dev-версия diffusers (у вас {ver}). "
            "Установите: pip install 'git+https://github.com/huggingface/diffusers.git'"
        )


# ==================== HELPER: BUILD DENOISE STEP GRAPH ====================

def _build_denoise_step(
    backbone,
    guidance,
    solver,
    *,
    use_cfg: bool = True,
    expose_num_steps: bool = False,
) -> ComputeGraph:
    """Build the inner per-step graph with explicit dual-pass CFG.
    
    Если use_cfg=True (default), создаёт два вызова backbone:
        backbone_cond   (с condition)  --|
                                        |--> guidance --> solver
        backbone_uncond (с null cond)  --|
    
    Никаких скрытых зависимостей — всё через порты и рёбра графа.
    
    Inputs (provided by LoopSubGraph at each iteration):
        latents:       current noisy latents
        timestep:      current timestep
        next_timestep: next timestep
        condition:     conditioner embedding
        uncond:        null conditioner embedding (for CFG)
    
    Outputs:
        next_latents:  denoised latents for next step
    """
    import copy
    
    step = ComputeGraph("denoise_step")
    step.add_node("backbone", backbone)
    step.add_node("guidance", guidance)
    step.add_node("solver", solver)
    
    if use_cfg:
        # Create a second backbone for unconditional pass.
        # Both share the SAME weights (same nn.Module object).
        # GraphExecutor calls process() which doesn't mutate state — safe.
        step.add_node("backbone_uncond", backbone)
        
        # Cond path: backbone -> guidance.model_output
        step.connect("backbone", "output", "guidance", "model_output")
        # Uncond path: backbone_uncond -> guidance.uncond_output
        step.connect("backbone_uncond", "output", "guidance", "uncond_output")
        
        # Guidance -> solver
        step.connect("guidance", "guided_output", "solver", "model_output")
        
        # Fan-out: latents -> all consumers
        step.expose_input("latents", "backbone", "x")
        step.expose_input("latents", "backbone_uncond", "x")
        step.expose_input("latents", "solver", "current_latents")
        
        # Fan-out: timestep -> all consumers
        step.expose_input("timestep", "backbone", "timestep")
        step.expose_input("timestep", "backbone_uncond", "timestep")
        step.expose_input("timestep", "solver", "timestep")
        
        # Condition -> cond backbone only
        step.expose_input("condition", "backbone", "condition")
        
        # Uncond -> uncond backbone only
        step.expose_input("uncond", "backbone_uncond", "condition")
    else:
        # No CFG — single pass
        step.connect("backbone", "output", "guidance", "model_output")
        step.connect("guidance", "guided_output", "solver", "model_output")
        
        step.expose_input("latents", "backbone", "x")
        step.expose_input("latents", "solver", "current_latents")
        
        step.expose_input("timestep", "backbone", "timestep")
        step.expose_input("timestep", "solver", "timestep")
        
        step.expose_input("condition", "backbone", "condition")
    
    step.expose_input("next_timestep", "solver", "next_timestep")
    if expose_num_steps:
        step.expose_input("num_steps", "solver", "num_steps")

    step.expose_output("next_latents", "solver", "next_latents")
    step.expose_output("latents", "solver", "next_latents")

    return step


def _build_denoise_step_batched_cfg(
    backbone_config: dict,
    guidance_config: dict,
    solver,
) -> ComputeGraph:
    """Build per-step graph with single batched UNet+CFG forward (diffusers parity).
    
    One backbone call with cat([latents, latents]) and cat([uncond_emb, cond_emb]),
    then chunk + CFG + optional guidance_rescale inside the block.
    """
    from yggdrasil.core.block.builder import BlockBuilder

    batched_config = {
        **backbone_config,
        "type": "backbone/unet2d_batched_cfg",
        "scale": guidance_config.get("scale", 7.5),
        "guidance_rescale": guidance_config.get("guidance_rescale", 0.0),
    }
    batched_backbone = BlockBuilder.build(batched_config)

    step = ComputeGraph("denoise_step")
    step.add_node("backbone", batched_backbone)
    step.add_node("solver", solver)

    step.connect("backbone", "output", "solver", "model_output")
    step.expose_input("latents", "backbone", "x")
    step.expose_input("latents", "solver", "current_latents")
    step.expose_input("timestep", "backbone", "timestep")
    step.expose_input("timestep", "solver", "timestep")
    step.expose_input("condition", "backbone", "condition")
    step.expose_input("uncond", "backbone", "uncond")
    step.expose_input("next_timestep", "solver", "next_timestep")
    step.expose_output("next_latents", "solver", "next_latents")
    step.expose_output("latents", "solver", "next_latents")

    return step


def _build_denoise_step_batched_cfg_euler(
    backbone_config: dict,
    guidance_config: dict,
    solver,
) -> ComputeGraph:
    """Batched CFG step with scale_model_input for EulerDiscreteScheduler (SDXL parity).
    
    Euler requires: scaled_latents = latents / (sigma^2+1)^0.5 before UNet.
    """
    from yggdrasil.core.block.builder import BlockBuilder

    batched_config = {
        **backbone_config,
        "type": "backbone/unet2d_batched_cfg",
        "scale": guidance_config.get("scale", 7.5),
        "guidance_rescale": guidance_config.get("guidance_rescale", 0.0),
    }
    batched_backbone = BlockBuilder.build(batched_config)
    scale_block = BlockBuilder.build({"type": "solver/scale_model_input"})
    scale_block.set_solver(solver)

    step = ComputeGraph("denoise_step")
    step.add_node("scale_input", scale_block)
    step.add_node("backbone", batched_backbone)
    step.add_node("solver", solver)

    step.expose_input("latents", "scale_input", "sample")
    step.expose_input("latents", "solver", "current_latents")
    step.expose_input("timestep", "scale_input", "timestep")
    step.expose_input("timestep", "backbone", "timestep")
    step.expose_input("timestep", "solver", "timestep")
    step.expose_input("num_steps", "scale_input", "num_steps")
    step.connect("scale_input", "scaled", "backbone", "x")
    step.connect("backbone", "output", "solver", "model_output")
    step.expose_input("condition", "backbone", "condition")
    step.expose_input("uncond", "backbone", "uncond")
    step.expose_input("image_prompt_embeds", "backbone", "image_prompt_embeds")
    step.expose_input("next_timestep", "solver", "next_timestep")
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
    use_batched_cfg: bool = False,
    use_euler_scale_input: bool = False,
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
    
    use_cfg = guidance_config.get("scale", 7.5) > 1.0
    
    graph = ComputeGraph(name)
    
    # ── Metadata for runtime parameter resolution ──
    graph.metadata = {
        "default_guidance_scale": guidance_config.get("scale", 7.5),
        "default_num_steps": num_steps,
        "default_width": default_width,
        "default_height": default_height,
        "latent_channels": codec_config.get("latent_channels", 4),
        "spatial_scale_factor": codec_config.get("spatial_scale_factor", 8),
        "init_noise_sigma": solver_config.get("init_noise_sigma", 1.0),
    }
    if use_euler_scale_input:
        graph.metadata["use_euler_init_sigma"] = True

    # Build blocks
    backbone = BlockBuilder.build(backbone_config)
    codec = BlockBuilder.build(codec_config)
    guidance = BlockBuilder.build(guidance_config)
    solver = BlockBuilder.build(solver_config)

    # Build inner step: Euler+scale (SD 1.5/SDXL parity), batched CFG, or dual-pass
    if use_cfg and use_batched_cfg and use_euler_scale_input:
        step_graph = _build_denoise_step_batched_cfg_euler(backbone_config, guidance_config, solver)
    elif use_cfg and use_batched_cfg:
        step_graph = _build_denoise_step_batched_cfg(backbone_config, guidance_config, solver)
    else:
        step_graph = _build_denoise_step(
            backbone, guidance, solver, use_cfg=use_cfg,
            expose_num_steps=(solver_config.get("type") == "solver/flow_euler"),
        )
    
    # Wrap in denoising loop. Schedule params live in solver (TZ §2.9: only solver, no separate scheduler).
    num_train_t = int(solver_config.get("num_train_timesteps") or schedule_config.get("num_train_timesteps", 1000))
    steps_offset = int(solver_config.get("steps_offset", 0))
    loop = LoopSubGraph.create(
        inner_graph=step_graph,
        num_iterations=num_steps,
        carry_vars=["latents"],
        num_train_timesteps=num_train_t,
        timestep_spacing="leading",
        steps_offset=steps_offset,
    )
    
    # Add conditioners
    conditioners = []
    for i, cond_config in enumerate(conditioner_configs):
        cond = BlockBuilder.build(cond_config)
        conditioners.append(cond)
        graph.add_node(f"conditioner_{i}", cond)
    
    # CFG: reuse the same conditioner for negative (one set of weights, two forward passes — like Diffusers)
    if use_cfg:
        graph.add_node("conditioner_negative", conditioners[0])
    
    # Position embedder: only add when wired (e.g. inside step graph). Currently not wired
    # at outer level (no timestep input), so adding it would cause executor to run it with
    # timestep=None and crash. Backbone (MMDiT) accepts position_embedding=None.
    # if position_config:
    #     position = BlockBuilder.build(position_config)
    #     graph.add_node("position", position)
    
    # Add loop and codec to outer graph
    graph.add_node("denoise_loop", loop)
    graph.add_node("codec", codec)
    
    # Wire conditioners — prompt input
    for i, cond in enumerate(conditioners):
        graph.expose_input(
            f"prompt_{i}" if i > 0 else "prompt",
            f"conditioner_{i}", "raw_condition",
        )
    
    # Condition path: conditioner -> loop.condition
    graph.connect("conditioner_0", "embedding", "denoise_loop", "condition")
    
    # Unconditional path: same CLIP with negative_prompt ("" = empty string, like diffusers)
    if use_cfg:
        graph.expose_input("negative_prompt", "conditioner_negative", "raw_condition")
        graph.connect("conditioner_negative", "embedding", "denoise_loop", "uncond")
    
    # Noise input -> loop
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
    
    Uses PNDMScheduler (как в diffusers StableDiffusionPipeline) для консистентности
    с output_diffusers.png при тех же prompt/seed/steps.
    
    Использование::
    
        graph = ComputeGraph.from_template("sd15_txt2img", device="cuda")
        outputs = graph.execute(prompt="a cat", guidance_scale=7.5, num_steps=28, seed=42)
        image = outputs["decoded"]
    """
    pretrained = kwargs.get("pretrained", "runwayml/stable-diffusion-v1-5")
    graph = _build_txt2img_graph(
        name="sd15_txt2img",
        backbone_config={"type": "backbone/unet2d_condition", "pretrained": pretrained, "fp16": True},
        codec_config={"type": "codec/autoencoder_kl", "pretrained": pretrained, "fp16": True, "scaling_factor": 0.18215, "latent_channels": 4, "spatial_scale_factor": 8},
        conditioner_configs=[{"type": "conditioner/clip_text", "pretrained": pretrained, "tokenizer_subfolder": "tokenizer", "text_encoder_subfolder": "text_encoder", "max_length": 77}],
        guidance_config={
            "type": "guidance/cfg",
            "scale": 7.5,
            "guidance_rescale": 0.7,
        },
        solver_config={
            "type": "solver/pndm_diffusers",
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "skip_prk_steps": True,
            "set_alpha_to_one": False,
            "steps_offset": 1,
        },
        schedule_config={"type": "noise/schedule/linear", "num_train_timesteps": 1000},
        process_config={"type": "diffusion/process/ddpm"},
        num_steps=kwargs.get("num_steps", 50),
        default_width=512,
        default_height=512,
        use_batched_cfg=True,
        use_euler_scale_input=False,
    )
    graph.metadata["use_scheduler_timesteps"] = True
    graph.metadata["base_model"] = "sd15"
    return graph


@register_template("sd15_txt2img_nobatch")
def sd15_txt2img_nobatch(**kwargs) -> ComputeGraph:
    """SD 1.5 txt2img с отдельными backbone/guidance/solver (без batched CFG).
    Подходит для add_controlnet_to_graph() — к этому графу можно добавить ControlNet."""
    pretrained = kwargs.get("pretrained", "runwayml/stable-diffusion-v1-5")
    return _build_txt2img_graph(
        name="sd15_txt2img_nobatch",
        backbone_config={"type": "backbone/unet2d_condition", "pretrained": pretrained, "fp16": True},
        codec_config={"type": "codec/autoencoder_kl", "pretrained": pretrained, "fp16": True, "scaling_factor": 0.18215, "latent_channels": 4, "spatial_scale_factor": 8},
        conditioner_configs=[{"type": "conditioner/clip_text", "pretrained": pretrained, "tokenizer_subfolder": "tokenizer", "text_encoder_subfolder": "text_encoder", "max_length": 77}],
        guidance_config={"type": "guidance/cfg", "scale": 7.5, "guidance_rescale": 0.7},
        solver_config={
            "type": "diffusion/solver/ddim",
            "eta": 0.0,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "num_train_timesteps": 1000,
            "clip_sample_range": 1.0,
            "steps_offset": 1,
        },
        schedule_config={"type": "noise/schedule/linear", "num_train_timesteps": 1000},
        process_config={"type": "diffusion/process/ddpm"},
        num_steps=kwargs.get("num_steps", 50),
        default_width=512,
        default_height=512,
        use_batched_cfg=False,
    )
    graph.metadata["base_model"] = "sd15"


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

def _build_sdxl_txt2img_graph(
    name: str,
    pretrained: str,
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    use_batched_cfg: bool = True,
    **kwargs,
) -> ComputeGraph:
    """Build SDXL txt2img with dual CLIP conditioner (encoder_hidden_states + added_cond_kwargs)."""
    from yggdrasil.core.block.builder import BlockBuilder
    from yggdrasil.core.graph.subgraph import LoopSubGraph

    graph = ComputeGraph(name)
    graph.metadata = {
        "default_guidance_scale": guidance_scale,
        "default_num_steps": num_steps,
        "default_width": 1024,
        "default_height": 1024,
        "latent_channels": 4,
        "spatial_scale_factor": 8,
        "modality": "image",
        "use_euler_init_sigma": True,  # init_noise_sigma taken from Euler solver at runtime
    }

    backbone_config = {"type": "backbone/unet2d_condition", "pretrained": pretrained, "fp16": True}
    codec_config = {
        "type": "codec/autoencoder_kl",
        "pretrained": pretrained,
        "fp16": True,
        "scaling_factor": 0.13025,
        "latent_channels": 4,
        "spatial_scale_factor": 8,
    }
    guidance_config = {
        "type": "guidance/cfg",
        "scale": guidance_scale,
        "guidance_rescale": 0.0,  # Diffusers default; use 0.7 for quality when not comparing
    }

    conditioner = BlockBuilder.build({
        "type": "conditioner/clip_sdxl",
        "pretrained": pretrained,
        "force_zeros_for_empty_prompt": True,
    })
    codec = BlockBuilder.build(codec_config)

    # SDXL in diffusers uses EulerDiscreteScheduler + scale_model_input; DDIM without scaling produces noise.
    solver_config = {
        "type": "solver/euler_discrete",
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "steps_offset": 1,
        "timestep_spacing": "leading",
    }
    solver = BlockBuilder.build(solver_config)

    if use_batched_cfg:
        step_graph = _build_denoise_step_batched_cfg_euler(backbone_config, guidance_config, solver)
    else:
        backbone = BlockBuilder.build(backbone_config)
        guidance = BlockBuilder.build(guidance_config)
        step_graph = _build_denoise_step(backbone, guidance, solver, use_cfg=True)

    loop = LoopSubGraph.create(
        inner_graph=step_graph,
        num_iterations=num_steps,
        carry_vars=["latents"],
        num_train_timesteps=1000,
        timestep_spacing="leading",
        steps_offset=1,
    )

    graph.add_node("conditioner", conditioner)
    graph.add_node("denoise_loop", loop)
    graph.add_node("codec", codec)

    graph.expose_input("prompt", "conditioner", "prompt")
    graph.expose_input("negative_prompt", "conditioner", "negative_prompt")
    graph.expose_input("height", "conditioner", "height")
    graph.expose_input("width", "conditioner", "width")
    graph.expose_input("latents", "denoise_loop", "initial_latents")
    graph.expose_input("timesteps", "denoise_loop", "timesteps")

    graph.connect("conditioner", "condition", "denoise_loop", "condition")
    graph.connect("conditioner", "uncond", "denoise_loop", "uncond")
    graph.connect("denoise_loop", "latents", "codec", "latent")

    graph.expose_output("decoded", "codec", "decoded")
    graph.expose_output("latents", "denoise_loop", "latents")
    return graph


def _default_generic_step_builder(
    metadata: Dict[str, Any],
    pretrained: str,
    num_steps: int,
    guidance_scale: float,
    **kwargs: Any,
) -> ComputeGraph:
    """Default step graph builder for template_id \"generic\" (SDXL batched CFG + Euler). Registered in StepBuilderRegistry."""
    from yggdrasil.core.block.builder import BlockBuilder
    backbone_config = {"type": "backbone/unet2d_condition", "pretrained": pretrained, "fp16": kwargs.get("fp16", True)}
    guidance_config = {"type": "guidance/cfg", "scale": guidance_scale, "guidance_rescale": 0.7}
    solver = BlockBuilder.build({
        "type": "solver/euler_discrete",
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "steps_offset": 1,
        "timestep_spacing": "leading",
    })
    return _build_denoise_step_batched_cfg_euler(backbone_config, guidance_config, solver)


def _step_sd3_builder(
    metadata: Dict[str, Any],
    pretrained: str,
    num_steps: int,
    guidance_scale: float,
    **kwargs: Any,
) -> ComputeGraph:
    """Step graph builder for SD3 (transformer + flow_euler + CFG)."""
    from yggdrasil.core.block.builder import BlockBuilder
    backbone = BlockBuilder.build({
        "type": "backbone/sd3_transformer",
        "pretrained": pretrained,
        "fp16": kwargs.get("fp16", True),
    })
    guidance = BlockBuilder.build({"type": "guidance/cfg", "scale": guidance_scale})
    solver = BlockBuilder.build({
        "type": "solver/flow_euler",
        "scheduler_pretrained": pretrained,
    })
    return _build_denoise_step(backbone, guidance, solver, use_cfg=True, expose_num_steps=True)


def _step_flux_builder(
    metadata: Dict[str, Any],
    pretrained: str,
    num_steps: int,
    guidance_scale: float,
    **kwargs: Any,
) -> ComputeGraph:
    """Step graph builder for Flux (transformer + flow_euler + CFG)."""
    from yggdrasil.core.block.builder import BlockBuilder
    backbone = BlockBuilder.build({
        "type": "backbone/flux_transformer",
        "pretrained": pretrained,
        "fp16": kwargs.get("fp16", True),
    })
    guidance = BlockBuilder.build({"type": "guidance/cfg", "scale": guidance_scale})
    solver = BlockBuilder.build({"type": "solver/flow_euler"})
    return _build_denoise_step(backbone, guidance, solver, use_cfg=True, expose_num_steps=True)


def _build_sdxl_denoise_loop_block(
    pretrained: str,
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> "LoopSubGraph":
    """Build SDXL denoise loop from pretrained (for use with add_node(type=\"loop/denoise_sdxl\", pretrained=...)).
    L1: metadata is used to select step template via get_step_template_id_for_metadata when provided at materialize.
    Dispatches via StepBuilderRegistry; \"generic\" is registered from this module.
    """
    from yggdrasil.core.block.builder import BlockBuilder
    from yggdrasil.core.graph.subgraph import LoopSubGraph
    from yggdrasil.core.graph.orchestrator import get_step_template_id_for_metadata, StepBuilderRegistry

    meta = metadata or {}
    template_id = get_step_template_id_for_metadata(meta)
    registry = StepBuilderRegistry()
    builder = registry.get(template_id) or registry.get("generic")
    if builder is not None:
        step_graph = builder(
            metadata=meta,
            pretrained=pretrained,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )
    else:
        backbone_config = {"type": "backbone/unet2d_condition", "pretrained": pretrained, "fp16": kwargs.get("fp16", True)}
        guidance_config = {"type": "guidance/cfg", "scale": guidance_scale, "guidance_rescale": 0.7}
        solver = BlockBuilder.build({
            "type": "solver/euler_discrete",
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "steps_offset": 1,
            "timestep_spacing": "leading",
        })
        step_graph = _build_denoise_step_batched_cfg_euler(backbone_config, guidance_config, solver)
    return LoopSubGraph.create(
        inner_graph=step_graph,
        num_iterations=num_steps,
        carry_vars=["latents"],
        num_train_timesteps=1000,
        timestep_spacing="leading",
        steps_offset=1,
    )


@register_block("loop/denoise_sdxl")
class DenoiseLoopSDXLBlock(AbstractBaseBlock):
    """SDXL denoise loop: один узел «цикл денойзинга» с Euler + batched CFG внутри.

    Существует чтобы add_node(type=\"backbone/unet2d_condition\", ...) мог подменяться на полный
    цикл (resolve_loop_for_backbone): пользователь добавляет «бэкбон», а в графе оказывается готовый
    loop с UNet, scale_model_input, solver и правильными timesteps/init_noise_sigma (parity с diffusers).
    Сборка через add_node(type=\"loop/denoise_sdxl\", pretrained=..., num_steps=..., guidance_scale=...)
    тоже поддерживается.
    """
    block_type = "loop/denoise_sdxl"

    def __init__(self, config: Dict | Any):
        from omegaconf import OmegaConf
        from yggdrasil.core.graph.subgraph import LoopSubGraph
        cfg = OmegaConf.create(config) if isinstance(config, dict) else config
        cfg_dict = OmegaConf.to_container(cfg, resolve=True) if hasattr(cfg, "to_container") else dict(cfg)
        # L1: graph passes metadata at materialize so step template is chosen by solver_type/modality
        graph_metadata = cfg_dict.pop("_graph_metadata", None)
        super().__init__(OmegaConf.create(cfg_dict) if cfg_dict else cfg)
        pretrained = cfg_dict.get("pretrained", "stabilityai/stable-diffusion-xl-base-1.0")
        num_steps = int(cfg_dict.get("num_steps", 50))
        guidance_scale = float(cfg_dict.get("guidance_scale", 7.5))
        self._loop: LoopSubGraph = _build_sdxl_denoise_loop_block(
            pretrained=pretrained, num_steps=num_steps, guidance_scale=guidance_scale,
            fp16=cfg_dict.get("fp16", True),
            metadata=graph_metadata,
        )

    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        from yggdrasil.core.graph.subgraph import LoopSubGraph
        return LoopSubGraph.declare_io()

    @property
    def num_iterations(self) -> int:
        return self._loop.num_iterations

    @num_iterations.setter
    def num_iterations(self, value: int) -> None:
        self._loop.num_iterations = value

    def process(self, **port_inputs: Any) -> Dict[str, Any]:
        return self._loop.process(**port_inputs)

    def parameters(self, recurse: bool = True):
        return self._loop.parameters(recurse=recurse)


# Register default step builders for StepBuilderRegistry (L1 extension).
def _register_default_step_builder() -> None:
    from yggdrasil.core.graph.orchestrator import StepBuilderRegistry
    reg = StepBuilderRegistry()
    reg.register("generic", _default_generic_step_builder)
    reg.register("step_sdxl", _default_generic_step_builder)
    reg.register("step_sd3", _step_sd3_builder)
    reg.register("step_flux", _step_flux_builder)


_register_default_step_builder()


@register_template("sdxl_txt2img")
def sdxl_txt2img(**kwargs) -> ComputeGraph:
    """Stable Diffusion XL text-to-image with dual text encoder (CLIP L + CLIP G) and added_cond_kwargs."""
    pretrained = kwargs.get("pretrained", "stabilityai/stable-diffusion-xl-base-1.0")
    graph = _build_sdxl_txt2img_graph(
        name="sdxl_txt2img",
        pretrained=pretrained,
        num_steps=kwargs.get("num_steps", 50),
        guidance_scale=kwargs.get("guidance_scale", 7.5),
        use_batched_cfg=True,
    )
    graph.metadata["base_model"] = "sdxl"
    return graph


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
    """Stable Diffusion 3 matching Diffusers StableDiffusion3Pipeline.

    Uses SD3Transformer2DModel (real weights), triple text encoder (CLIP+CLIP2+T5),
    and FlowMatchEulerDiscrete-style solver so output matches diffusers.png.
    Repo: stabilityai/stable-diffusion-3-medium-diffusers.
    """
    pretrained = kwargs.get("pretrained", "stabilityai/stable-diffusion-3-medium-diffusers")
    graph = _build_txt2img_graph(
        name="sd3_txt2img",
        backbone_config={"type": "backbone/sd3_transformer", "pretrained": pretrained, "fp16": True},
        codec_config={"type": "codec/autoencoder_kl", "pretrained": pretrained, "fp16": True, "latent_channels": 16},
        conditioner_configs=[
            {"type": "conditioner/sd3_text", "pretrained": pretrained, "fp16": True},
        ],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 5.0)},
        solver_config={"type": "solver/flow_euler", "scheduler_pretrained": pretrained},
        schedule_config={"type": "noise/schedule/sigmoid", "num_train_timesteps": 1000},
        process_config={"type": "diffusion/process/flow/rectified"},
        num_steps=kwargs.get("num_steps", 28),
        default_width=1024,
        default_height=1024,
    )
    graph.metadata["base_model"] = "sd3"
    graph.metadata["pretrained"] = pretrained
    graph.metadata["latent_channels"] = 16
    graph.metadata["spatial_scale_factor"] = 8
    return graph


@register_template("sd3_img2img")
def sd3_img2img(**kwargs) -> ComputeGraph:
    """Stable Diffusion 3 image-to-image."""
    graph = sd3_txt2img(**kwargs)
    graph.name = "sd3_img2img"
    graph.expose_input("source_image", "codec", "pixel_data")
    return graph


# ==================== FLUX ====================

@register_template("flux_txt2img")
def flux_txt2img(**kwargs) -> ComputeGraph:
    """FLUX.1 text-to-image with MMDiT and rectified flow."""
    graph = _build_txt2img_graph(
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
    graph.metadata["base_model"] = "flux"
    return graph


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
    prior_step = _build_denoise_step(prior_backbone, prior_guidance, prior_solver, use_cfg=True)
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
        step = _build_denoise_step(bb, g, s, use_cfg=True)
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


# ==================== FLUX.2 ====================

def _flux2_base(name: str, variant: str, **kwargs) -> ComputeGraph:
    """Base builder for all FLUX.2 variants.

    FLUX.2 architecture (diffusers v0.37.0.dev0):
        - Backbone: Flux2Transformer2DModel
        - Text encoder: Mistral3 (dev/schnell) или Qwen (klein), joint_attention_dim 15360 / 7680
        - VAE: dev/schnell — AutoencoderKL из репо; Klein — свой AutoencoderKLFlux2 (subfolder vae)
        - Scheduler: FlowMatchEulerDiscrete
        - Guidance: embedded (guidance-distilled) или false для Klein
    """
    if variant == "klein":
        _require_diffusers_klein()
    pretrained = kwargs.get("pretrained", f"black-forest-labs/FLUX.2-{variant}")
    token = kwargs.get("token")
    backbone_cfg = {
        "type": "backbone/flux2_transformer",
        "variant": variant,
        "pretrained": pretrained,
        "hidden_dim": 3072,
        "num_layers": 8,
        "num_heads": 48,
        "in_channels": 128,
        "bf16": True,
    }
    if token is not None:
        backbone_cfg["token"] = token
    # Klein использует свой VAE: AutoencoderKLFlux2 (не AutoencoderKL)
    if variant == "klein":
        codec_cfg = {
            "type": "codec/autoencoder_kl_flux2",
            "pretrained": pretrained,
            "subfolder": "vae",
            "bf16": True,
            "latent_channels": 32,
            "spatial_scale_factor": 16,
        }
    else:
        codec_cfg = {
            "type": "codec/autoencoder_kl",
            "pretrained": pretrained,
            "latent_channels": 32,
            "spatial_scale_factor": 16,
        }
    if token is not None:
        codec_cfg["token"] = token
    # Klein uses Qwen text encoder (same repo, subfolder text_encoder), joint_attention_dim=7680
    if variant == "klein":
        conditioner_cfgs = [
            {
                "type": "conditioner/qwen_causal",
                "pretrained": pretrained,
                "subfolder": "text_encoder",
                "max_length": 512,
                "hidden_layers": [10, 20, 30],
                "embedding_dim": 7680,
            },
        ]
    else:
        conditioner_cfgs = [
            {
                "type": "conditioner/mistral3",
                "pretrained": kwargs.get("text_encoder", "mistralai/Mistral-Small-3.1-24B-Instruct-2503"),
                "max_length": 512,
                "hidden_layers": [10, 20, 30],
                "embedding_dim": 15360,
            },
        ]
    if token is not None:
        conditioner_cfgs[0]["token"] = token
    return _build_txt2img_graph(
        name=name,
        backbone_config=backbone_cfg,
        codec_config=codec_cfg,
        conditioner_configs=conditioner_cfgs,
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 4.0)},
        solver_config={"type": "solver/flow_euler"},
        schedule_config={"type": "noise/schedule/sigmoid"},
        process_config={"type": "diffusion/process/flow/rectified"},
        num_steps=kwargs.get("num_steps", 50),
        default_width=1024,
        default_height=1024,
    )


@register_template("flux2_txt2img")
def flux2_txt2img(**kwargs) -> ComputeGraph:
    """FLUX.2 dev text-to-image."""
    return _flux2_base("flux2_txt2img", "dev", **kwargs)


@register_template("flux2_schnell")
def flux2_schnell(**kwargs) -> ComputeGraph:
    """FLUX.2 Schnell — fast 4-step generation."""
    kwargs.setdefault("num_steps", 4)
    kwargs.setdefault("guidance_scale", 0.0)  # Schnell: no CFG
    return _flux2_base("flux2_schnell", "schnell", **kwargs)


@register_template("flux2_fill")
def flux2_fill(**kwargs) -> ComputeGraph:
    """FLUX.2 Fill — inpainting/outpainting."""
    graph = _flux2_base("flux2_fill", "Fill-dev", **kwargs)
    graph.expose_input("mask", "denoise_loop", "mask")
    graph.expose_input("source_image", "codec", "pixel_data")
    return graph


@register_template("flux2_canny")
def flux2_canny(**kwargs) -> ComputeGraph:
    """FLUX.2 Canny — edge-conditioned generation."""
    graph = _flux2_base("flux2_canny", "Canny-dev", **kwargs)
    graph.expose_input("canny_image", "denoise_loop", "image_condition")
    return graph


@register_template("flux2_depth")
def flux2_depth(**kwargs) -> ComputeGraph:
    """FLUX.2 Depth — depth-conditioned generation."""
    graph = _flux2_base("flux2_depth", "Depth-dev", **kwargs)
    graph.expose_input("depth_map", "denoise_loop", "image_condition")
    return graph


@register_template("flux2_redux")
def flux2_redux(**kwargs) -> ComputeGraph:
    """FLUX.2 Redux — image variation."""
    graph = _flux2_base("flux2_redux", "Redux-dev", **kwargs)
    graph.expose_input("reference_image", "denoise_loop", "image_condition")
    return graph


@register_template("flux2_kontext")
def flux2_kontext(**kwargs) -> ComputeGraph:
    """FLUX.2 Kontext — multi-image context generation."""
    graph = _flux2_base("flux2_kontext", "Kontext-dev", **kwargs)
    graph.expose_input("context_images", "denoise_loop", "image_condition")
    return graph


@register_template("flux2_klein")
def flux2_klein(**kwargs) -> ComputeGraph:
    """FLUX.2 [klein] — быстрая генерация (4 шага, 9B/4B)."""
    kwargs.setdefault("num_steps", 4)
    kwargs.setdefault("guidance_scale", 0.0)
    kwargs.setdefault("pretrained", "black-forest-labs/FLUX.2-klein-9B")
    return _flux2_base("flux2_klein", "klein", **kwargs)


# ==================== Wan 2.1 (Video) ====================

def _wan_base(name: str, pretrained: str, **kwargs) -> ComputeGraph:
    """Base builder for Wan video pipelines."""
    return _build_txt2img_graph(
        name=name,
        backbone_config={
            "type": "backbone/wan_transformer",
            "pretrained": pretrained,
            "hidden_dim": kwargs.get("hidden_dim", 1536),
            "in_channels": 16,
            "fp16": True,
        },
        codec_config={
            "type": "codec/wan_vae",
            "pretrained": pretrained,
            "latent_channels": 16,
            "spatial_scale_factor": 8,
        },
        conditioner_configs=[
            {"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14", "max_length": 77, "embedding_dim": 768},
            {"type": "conditioner/t5_text", "pretrained": "google/umt5-xxl", "max_length": 512},
        ],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 5.0)},
        solver_config={"type": "solver/euler"},
        schedule_config={"type": "noise/schedule/linear"},
        process_config={"type": "diffusion/process/flow/rectified"},
        num_steps=kwargs.get("num_steps", 50),
        default_width=kwargs.get("default_width", 720),
        default_height=kwargs.get("default_height", 480),
    )


@register_template("wan_t2v")
def wan_t2v(**kwargs) -> ComputeGraph:
    """Wan 2.1 text-to-video."""
    pretrained = kwargs.get("pretrained", "Wan-AI/Wan2.1-T2V-14B-Diffusers")
    graph = _wan_base("wan_t2v", pretrained, **kwargs)
    graph.metadata["num_frames"] = kwargs.get("num_frames", 81)
    graph.metadata["output_type"] = "video"
    return graph


@register_template("wan_i2v")
def wan_i2v(**kwargs) -> ComputeGraph:
    """Wan 2.1 image-to-video."""
    pretrained = kwargs.get("pretrained", "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers")
    from yggdrasil.core.block.builder import BlockBuilder
    
    graph = _wan_base("wan_i2v", pretrained, **kwargs)
    
    # Add CLIP vision conditioner for reference image
    clip_vision = BlockBuilder.build({
        "type": "conditioner/clip_vision",
        "pretrained": "openai/clip-vit-large-patch14",
    })
    graph.add_node("image_conditioner", clip_vision)
    graph.expose_input("reference_image", "image_conditioner", "raw_condition")
    
    graph.metadata["num_frames"] = kwargs.get("num_frames", 81)
    graph.metadata["output_type"] = "video"
    return graph


@register_template("wan_flf2v")
def wan_flf2v(**kwargs) -> ComputeGraph:
    """Wan 2.1 first+last frame to video."""
    pretrained = kwargs.get("pretrained", "Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers")
    graph = _wan_base("wan_flf2v", pretrained, **kwargs)
    graph.expose_input("first_frame", "denoise_loop", "image_condition")
    graph.expose_input("last_frame", "denoise_loop", "last_frame")
    graph.metadata["num_frames"] = kwargs.get("num_frames", 81)
    graph.metadata["output_type"] = "video"
    return graph


@register_template("wan_fun_control")
def wan_fun_control(**kwargs) -> ComputeGraph:
    """Wan 2.1 Fun Control — controllable video generation."""
    pretrained = kwargs.get("pretrained", "Wan-AI/Wan2.1-Fun-Control-14B-Diffusers")
    graph = _wan_base("wan_fun_control", pretrained, **kwargs)
    graph.expose_input("control_video", "denoise_loop", "image_condition")
    graph.metadata["num_frames"] = kwargs.get("num_frames", 81)
    graph.metadata["output_type"] = "video"
    return graph


# ==================== QwenImage ====================

@register_template("qwen_image_txt2img")
def qwen_image_txt2img(**kwargs) -> ComputeGraph:
    """QwenImage text-to-image generation."""
    return _build_txt2img_graph(
        name="qwen_image_txt2img",
        backbone_config={
            "type": "backbone/qwen_image",
            "pretrained": kwargs.get("pretrained", "Qwen/QwenImage-1.5B"),
            "in_channels": 4,
            "hidden_dim": 320,
            "fp16": True,
        },
        codec_config={
            "type": "codec/autoencoder_kl",
            "latent_channels": 4,
            "spatial_scale_factor": 8,
        },
        conditioner_configs=[
            {"type": "conditioner/qwen_vl", "pretrained": "Qwen/Qwen2.5-VL-3B-Instruct", "embedding_dim": 2048},
        ],
        guidance_config={"type": "guidance/cfg", "scale": kwargs.get("guidance_scale", 7.5)},
        solver_config={"type": "solver/euler"},
        schedule_config={"type": "noise/schedule/linear"},
        process_config={"type": "diffusion/process/ddpm"},
        num_steps=kwargs.get("num_steps", 50),
        default_width=512,
        default_height=512,
    )


@register_template("qwen_image_edit")
def qwen_image_edit(**kwargs) -> ComputeGraph:
    """QwenImage image editing (text-guided)."""
    graph = qwen_image_txt2img(**kwargs)
    graph.name = "qwen_image_edit"
    graph.expose_input("source_image", "codec", "pixel_data")
    graph.expose_input("edit_mask", "denoise_loop", "edit_mask")
    return graph
