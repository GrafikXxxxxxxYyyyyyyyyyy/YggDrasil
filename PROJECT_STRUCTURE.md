# YggDrasil — Project Structure and File Descriptions

This document describes the full structure of the YggDrasil project: every file, its purpose, role, and implementation approach.

**YggDrasil** is a universal Lego-style diffusion framework supporting any modality (images, video, audio) and any model (SD 1.5, SDXL, SD3, FLUX, etc.). All components are composable blocks connected via typed ports in a directed acyclic graph (ComputeGraph).

---

## Root Files

### `pyproject.toml`
**Purpose:** Project metadata, dependencies, and build configuration.

**Role:** Defines the Python package `yggdrasil`, its version (0.1.0), required dependencies (torch, diffusers, transformers, fastapi, gradio), optional extras (`full`, `klein`, `train`, `dev`), and Ruff linting rules.

**Implementation:** Uses PEP 621 format. Entry point `yggdrasil` maps to `yggdrasil.cli:main`.

---

### `setup.py`
**Purpose:** Legacy setup script for editable installs.

**Role:** Allows `pip install -e .`; delegates to setuptools and pyproject.toml.

---

### `README.md`
**Purpose:** User-facing project overview, install instructions, and CLI usage.

**Role:** Quick start for installation, commands (`yggdrasil ui`, `yggdrasil api`), and optional dependency groups.

---

### `TECHNICAL_SPECIFICATION.md`
**Purpose:** Formal technical specification (ТЗ v1.2) of the architecture.

**Role:** Defines acceptance criteria, implementation phases, block hierarchy, graph model, and unified API (InferencePipeline, TrainingPipeline, from_config, from_pretrained, etc.).

---

### `REFACTORING_STATUS.md`
**Purpose:** Refactoring completion report.

**Role:** Confirms all acceptance criteria from ТЗ v1.2 are met; lists main files and phases 1–5 status.

---

### `AUDIT_REPORT.md`
**Purpose:** Codebase audit report (if present).

**Role:** Quality and compliance notes.

---

## `yggdrasil/` — Main Package

### `yggdrasil/__init__.py`
**Purpose:** Package entry, version, and high-level API.

**Role:** Exports `InferencePipeline`, `TrainingPipeline`, `load_model`, `generate`, `list_presets`, `list_blocks`. Applies TensorFlow/Metal compatibility patch; runs block auto-discovery on import.

**Implementation:** `load_model()` resolves preset, YAML path, or HuggingFace ID; `generate()` wraps `DiffusionSampler` for simple generation.

---

### `yggdrasil/__main__.py`
**Purpose:** Allow running `python -m yggdrasil`.

**Role:** Forwards to CLI entry point.

---

### `yggdrasil/cli.py`
**Purpose:** Command-line interface.

**Role:** Parses `ui`, `api`, `serve`, `generate`, `train`, etc., and invokes the corresponding handlers (Gradio, FastAPI, etc.).

---

### `yggdrasil/pipeline.py`
**Purpose:** High-level InferencePipeline and TrainingPipeline API (diffusers-like).

**Role:** One-line loading (`InferencePipeline.from_pretrained`, `from_template`, `from_diffusers`, `from_combined`), generation, training, and checkpointing. Internally wraps a `ComputeGraph`.

**Implementation:** `InferencePipeline` holds a graph; `__call__` runs `GraphExecutor` with prompt/image inputs and returns `PipelineOutput` (images, latents, raw).

---

### `yggdrasil/runner.py`
**Purpose:** Workflow runner for combined pipelines.

**Role:** Executes multi-stage workflows (e.g. T2I → img2img) from YAML config; `Runner.execute()` runs stages in sequence.

---

### `yggdrasil/hub.py`
**Purpose:** Model registry and resolution.

**Role:** Maps HuggingFace model IDs to template names and kwargs; used by `InferencePipeline.from_pretrained` for native YggDrasil graphs.

---

## `yggdrasil/core/` — Core Framework

### `yggdrasil/core/__init__.py`
**Purpose:** Re-exports core submodules.

---

## `yggdrasil/core/block/` — Block System

### `yggdrasil/core/block/base.py`
**Purpose:** Base class for all Lego blocks (`AbstractBaseBlock`).

**Role:** Declares `declare_io()` and `process(**port_inputs)`. Handles slots, child attachment via config, pre/post hooks. Legacy `_forward_impl` / `_define_slots` still supported.

**Implementation:** Subclasses implement `declare_io()` and `process()`; `_build_slots()` uses `BlockBuilder` to recursively build children from config.

---

### `yggdrasil/core/block/registry.py`
**Purpose:** Global block registry and auto-discovery.

**Role:** `@register_block("category/name")` registers block classes; `BlockRegistry.get()`, `list_blocks()`; auto-import from `yggdrasil.blocks` and `yggdrasil.plugins`. Suggests similar block names on KeyError.

**Implementation:** Walks packages with `pkgutil.walk_packages`, logs import errors instead of swallowing them.

---

### `yggdrasil/core/block/builder.py`
**Purpose:** Recursive block construction from config (YAML/dict).

**Role:** `BlockBuilder.build(config)` → instantiated `AbstractBaseBlock`; resolves `type` via registry and fills slots from nested config.

---

### `yggdrasil/core/block/slot.py`
**Purpose:** Slot definition for attaching blocks (Lego socket).

**Role:** `Slot(name, accepts, multiple, optional)` describes what block types a slot accepts; `check_compatible(block)` validates attachment.

**Implementation:** `accepts` can be class, tuple of classes, or prefix string (e.g. `"backbone/"`).

---

### `yggdrasil/core/block/port.py`
**Purpose:** Typed I/O ports for dataflow.

**Role:** `InputPort`, `OutputPort`, `TensorSpec` define block interfaces; used for graph validation and executor wiring. `TensorSpec` encodes ndim, channels, space (latent/pixel/embedding), modality.

**Implementation:** `Port` has name, direction, data_type, optional, multiple; `TensorSpec.is_compatible()` for connection checks.

---

### `yggdrasil/core/block/graph.py`
**Purpose:** Block-level graph (legacy `BlockGraph`).

**Role:** Visualizes and validates slot-based dependencies between blocks (before ComputeGraph).

---

## `yggdrasil/core/diffusion/` — Diffusion Process

### `yggdrasil/core/diffusion/process.py`
**Purpose:** Abstract diffusion process (DDPM, flow matching, consistency, etc.).

**Role:** `AbstractDiffusionProcess` defines `forward_process()`, `reverse_step()`, `predict_x0()`, `predict_velocity()`. Optional slot for `NoiseSchedule`.

**Implementation:** Concrete classes: `DDPMProcess`, `RectifiedFlowProcess`, etc.

---

### `yggdrasil/core/diffusion/ddpm.py`
**Purpose:** DDPM forward/reverse process.

**Role:** Implements the DDPM formulation (add noise, predict noise, reverse step).

---

### `yggdrasil/core/diffusion/flow.py`
**Purpose:** Flow-matching processes (Rectified Flow, EDM, etc.).

**Role:** Implements ODE-based diffusion with velocity prediction.

---

### `yggdrasil/core/diffusion/consistency.py`
**Purpose:** Consistency model process.

**Role:** Single-step consistency mapping from noise to data.

---

### `yggdrasil/core/diffusion/noise/schedule.py`
**Purpose:** Noise schedule (linear, cosine, sigmoid).

**Role:** Maps timestep to alpha/ sigma for diffusion; used by process and solvers.

---

### `yggdrasil/core/diffusion/noise/sampler.py`
**Purpose:** Timestep and noise sampling.

**Role:** Samples `t` and optional `noise` for training and inference.

---

### `yggdrasil/core/diffusion/solver/base.py`
**Purpose:** Abstract solver for reverse diffusion steps.

**Role:** `AbstractSolver` defines one-step denoising; takes `model_output`, `current_latents`, `timestep`, returns `next_latents`.

---

### `yggdrasil/core/diffusion/solver/ddim.py`
**Purpose:** DDIM solver.

**Role:** Deterministic reverse step with eta parameter.

---

### `yggdrasil/core/diffusion/solver/heun.py`
**Purpose:** Heun (second-order) ODE solver.

**Role:** Used by flow-matching models (e.g. FLUX).

---

### `yggdrasil/core/diffusion/solver/custom_ode.py`
**Purpose:** Generic ODE step integration.

**Role:** Allows custom step functions for research.

---

## `yggdrasil/core/engine/` — Sampling Engine

### `yggdrasil/core/engine/sampler.py`
**Purpose:** Universal diffusion sampler.

**Role:** Orchestrates model, diffusion process, solver, guidance. Supports slot-based (legacy) and graph-based execution. Slots: `model`, `diffusion_process`, `solver`, `noise_schedule`.

**Implementation:** Iterates timesteps, calls backbone (with CFG), solver, noise schedule; optional progress bar and step hooks.

---

### `yggdrasil/core/engine/pipeline.py`
**Purpose:** Abstract pipeline (legacy).

**Role:** Base for `train_step`, `infer_step`, `save`; used by engine orchestration.

---

### `yggdrasil/core/engine/loop.py`
**Purpose:** Sampling loop with hooks.

**Role:** Step-level callbacks for FaceDetailer, ControlNet, etc.

---

### `yggdrasil/core/engine/state.py`
**Purpose:** Diffusion state container.

**Role:** `DiffusionState` holds latents, timestep, condition, cache; used for streaming and continuation.

---

## `yggdrasil/core/graph/` — Compute Graph

### `yggdrasil/core/graph/graph.py`
**Purpose:** `ComputeGraph` — DAG of blocks with typed ports.

**Role:** Main structure for pipelines. Nodes = blocks; edges = port connections. `add_node()`, `connect()`, `add_node(type=..., auto_connect=True)`, `add_stage()`, `expose_input()`, `expose_output()`, `to(device)`, `from_template()`, `from_yaml()`, `to_yaml()`.

**Implementation:** Topological sort for execution; fan-out from graph inputs; device/dtype propagation.

---

### `yggdrasil/core/graph/role_rules.py`
**Purpose:** Auto-connection rules for `add_node(type=...)`.

**Role:** Maps `block_type` prefix to role (backbone, codec, conditioner, solver, adapter, etc.); `get_connection_rules(role)` returns target node/port for auto-wiring. `schedule/` maps to solver (merged).

---

### `yggdrasil/core/graph/stage.py`
**Purpose:** `AbstractStage` — one pipeline stage as a graph node.

**Role:** Stage = inner `ComputeGraph`; `AbstractStage` is a node in the outer pipeline. Detailer, Upscaler, Prior are stages. `process()` runs inner graph via `GraphExecutor`.

**Implementation:** Delegates to `GraphExecutor().execute(self.graph, **port_inputs)`.

---

### `yggdrasil/core/graph/executor.py`
**Purpose:** Executes `ComputeGraph`.

**Role:** Topological sort, gather inputs from cache and graph inputs, call `block.process(**inputs)`, cache outputs. Supports `execute_training()` (with gradients), callbacks, strict port validation, optional cache invalidation.

---

### `yggdrasil/core/graph/compiler.py`
**Purpose:** Graph compilation / optimization (if present).

**Role:** Preprocessing before execution (e.g. fusion, constant folding).

---

### `yggdrasil/core/graph/adapters.py`
**Purpose:** Adapter utilities for graph construction.

**Role:** Helpers for ControlNet, T2I-Adapter, LoRA integration in graphs.

---

### `yggdrasil/core/graph/subgraph.py`
**Purpose:** Subgraph / loop subgraph nodes.

**Role:** `LoopSubGraph` for denoising loop; iterates and feeds step graph with latents, timestep, condition.

---

### `yggdrasil/core/graph/nodes/conditional.py`
**Purpose:** Conditional execution nodes.

**Role:** Branch execution based on condition (e.g. CFG on/off).

---

### `yggdrasil/core/graph/nodes/for_loop.py`
**Purpose:** For-loop node for iterative execution.

**Role:** Repeats a subgraph N times (e.g. denoising steps).

---

### `yggdrasil/core/graph/nodes/parallel.py`
**Purpose:** Parallel execution node.

**Role:** Runs multiple branches in parallel (e.g. cond + uncond for CFG).

---

### `yggdrasil/core/graph/templates/__init__.py`
**Purpose:** Template registry and `from_template()`.

**Role:** Registers templates (sd15_txt2img, sdxl_txt2img, etc.) and builds `ComputeGraph` from name + kwargs.

---

### `yggdrasil/core/graph/templates/image_pipelines.py`
**Purpose:** Image diffusion templates (SD 1.5, SDXL, SD3, FLUX, DiT, etc.).

**Role:** `sd15_txt2img()`, `sdxl_txt2img()`, `sd3_txt2img()`, `flux_txt2img()`, etc. Each returns a full `ComputeGraph` with denoise loop, CFG, codec, conditioner. `_build_denoise_step()` creates the per-step graph with dual backbone for CFG.

---

### `yggdrasil/core/graph/templates/control_pipelines.py`
**Purpose:** ControlNet / T2I-Adapter templates.

**Role:** Templates that add adapter nodes feeding `backbone.adapter_features`.

---

### `yggdrasil/core/graph/templates/video_pipelines.py`
**Purpose:** Video diffusion templates (CogVideoX, AnimateDiff, etc.).

**Role:** 3D UNet / video transformer graphs with temporal conditioning.

---

### `yggdrasil/core/graph/templates/audio_pipelines.py`
**Purpose:** Audio diffusion templates (Stable Audio, etc.).

**Role:** 1D UNet / EnCodec graphs.

---

### `yggdrasil/core/graph/templates/training_pipelines.py`
**Purpose:** Training graph templates.

**Role:** Graphs with `DatasetBlock`, `LossBlock`, trainable nodes.

---

### `yggdrasil/core/graph/templates/animatediff_extensions.py`
**Purpose:** AnimateDiff-specific extensions.

**Role:** Motion modules, temporal attention in video graphs.

---

### `yggdrasil/core/graph/templates/specialized_pipelines.py`
**Purpose:** Specialized pipelines (Kandinsky, DeepFloyd, etc.).

**Role:** Multi-stage or model-specific templates.

---

## `yggdrasil/core/model/` — Model Abstractions

### `yggdrasil/core/model/modular.py`
**Purpose:** `ModularDiffusionModel` — the main model container.

**Role:** Composes backbone, codec, conditioner(s), guidance via slots; single `forward()` / `process()` entry. Used by legacy slot-based sampler.

---

### `yggdrasil/core/model/backbone.py`
**Purpose:** `AbstractBackbone` — UNet, DiT, Transformer, etc.

**Role:** Ports: x, timestep, condition, position_embedding, adapter_features → output. All backbones inherit this.

---

### `yggdrasil/core/model/codec.py`
**Purpose:** `AbstractLatentCodec` — VAE encode/decode.

**Role:** Pixel ↔ latent; `encode()`, `decode()`; scaling factor.

---

### `yggdrasil/core/model/conditioner.py`
**Purpose:** `AbstractConditioner` — text, image, multi-modal.

**Role:** Input (text, image) → embedding dict; ports for text, image, output.

---

### `yggdrasil/core/model/guidance.py`
**Purpose:** `AbstractGuidance` — CFG, PAG, FreeU, etc.

**Role:** Combines conditional and unconditional model outputs; `model_output`, `uncond_output` → `guided_output`.

---

### `yggdrasil/core/model/position.py`
**Purpose:** `AbstractPositionEmbedder` — sinusoidal, RoPE, learned.

**Role:** Timestep / position → embedding for backbone.

---

### `yggdrasil/core/model/postprocess.py`
**Purpose:** Post-processing (e.g. VAE decode, rescale).

**Role:** Latents → final output (pixel, audio, etc.).

---

### `yggdrasil/core/model/inner_module.py`
**Purpose:** Inner modules (e.g. ControlNet) feeding backbone.

**Role:** Adapter-like blocks that inject into backbone; `adapter_features` port.

---

### `yggdrasil/core/model/outer_module.py`
**Purpose:** Outer preprocessing modules.

**Role:** Input → processed input for next stage.

---

### `yggdrasil/core/model/processor.py`
**Purpose:** `AbstractProcessor` — pre/post-processing (resize, normalize, crop).

**Role:** Raw data → tensor; tensor → final format. Input/output ports.

---

## `yggdrasil/core/unified/` — Unified Contract

### `yggdrasil/core/unified/contract.py`
**Purpose:** Unified API contract for all modalities.

**Role:** Standard input/output schemas for pipelines.

---

### `yggdrasil/core/unified/modality.py`
**Purpose:** Modality definitions (image, audio, video, etc.).

**Role:** Enum/constants for modality handling.

---

### `yggdrasil/core/unified/steps.py`
**Purpose:** Step definitions for unified workflows.

**Role:** Common step types (encode, denoise, decode, etc.).

---

## `yggdrasil/core/utils/` — Utilities

### `yggdrasil/core/utils/config.py`
**Purpose:** OmegaConf helpers, validation, slot config.

---

### `yggdrasil/core/utils/tensor.py`
**Purpose:** `DiffusionTensor` — tensor with metadata (space, modality).

---

### `yggdrasil/core/utils/hooks.py`
**Purpose:** Pre/post hooks for blocks.

**Role:** `register_pre_hook`, `register_post_hook`; called before/after `process()`.

---

## `yggdrasil/blocks/` — Concrete Blocks

### `yggdrasil/blocks/backbones/`
**Purpose:** Concrete backbone implementations.

| File | Purpose |
|------|---------|
| `unet_2d_condition.py` | SD 1.5 / SDXL UNet |
| `unet_3d_condition.py` | Video UNet |
| `flux_transformer.py` | FLUX Transformer |
| `flux2_transformer.py` | FLUX.2 [klein] |
| `sd3_transformer.py` | SD3 Transformer |
| `dit.py` | DiT (PixArt, Latte, etc.) |
| `transformer_2d.py` | Generic 2D Transformer |
| `mmdit.py` | Hunyuan MMDiT |
| `wan_transformer.py` | Wan video transformer |
| `unet2d_batched_cfg.py` | Batched CFG UNet |
| `qwen_image.py` | Qwen image backbone |
| `equivariant_gnn.py` | Molecular GNN |

---

### `yggdrasil/blocks/codecs/`
**Purpose:** VAE / codec implementations.

| File | Purpose |
|------|---------|
| `autoencoder_kl.py` | SD/FLUX VAE (AutoencoderKL) |
| `autoencoder_kl_flux2.py` | FLUX.2 VAE |
| `encodec.py` | Audio EnCodec |
| `vqgan.py` | VQ-GAN |
| `wan_vae.py` | Wan VAE |
| `identity.py` | Identity (no encode/decode) |

---

### `yggdrasil/blocks/conditioners/`
**Purpose:** Text and multi-modal encoders.

| File | Purpose |
|------|---------|
| `clip_text.py` | CLIP text encoder |
| `clip_sdxl.py` | SDXL dual text encoder |
| `clip_vision.py` | CLIP vision encoder |
| `t5_text.py` | T5 text encoder |
| `sd3_text.py` | SD3 T5 + CLIP |
| `mistral3.py` | Mistral3 text |
| `qwen_causal.py` | Qwen causal |
| `qwen_vl.py` | Qwen-VL |
| `multi_modal.py` | Multi-modal fusion |
| `image_encoder.py` | Image encoder |
| `clap.py` | CLAP (audio-text) |
| `prompt_schedule.py` | Prompt scheduling |
| `null.py` | Null/unconditional |

---

### `yggdrasil/blocks/adapters/`
**Purpose:** ControlNet, LoRA, T2I-Adapter, etc.

| File | Purpose |
|------|---------|
| `base.py` | Abstract adapter |
| `controlnet.py` | ControlNet |
| `t2i_adapter.py` | T2I-Adapter |
| `lora.py` | LoRA |
| `ip_adapter.py` | IP-Adapter |
| `fusion.py` | Fusion adapter |
| `hypernetwork.py` | Hypernetwork |
| `textual_inversion.py` | Textual Inversion |

---

### `yggdrasil/blocks/guidances/`
**Purpose:** CFG, FreeU, PAG, SAG.

| File | Purpose |
|------|---------|
| `cfg.py` | Classifier-free guidance |
| `freeu.py` | FreeU |
| `pag.py` | Prompt-adapter guidance |
| `sag.py` | Self-attention guidance |

---

### `yggdrasil/blocks/solvers/`
**Purpose:** Discrete/continuous solvers.

| File | Purpose |
|------|---------|
| `euler_discrete.py` | Euler discrete |
| `flow_euler.py` | Flow Euler |
| `euler.py` | Euler (generic) |
| `dpm.py` | DPM-solver |
| `deis.py` | DEIS |
| `lms.py` | LMS |
| `pndm.py` | PNDM |
| `pndm_diffusers.py` | PNDM (diffusers-compat) |
| `unipc.py` | UniPC |
| `scale_model_input.py` | Sigma scaling (EDM, etc.) |

---

### `yggdrasil/blocks/schedules/`
**Purpose:** Diffusion schedules (merged into solver config).

| File | Purpose |
|------|---------|
| `diffusion_schedule.py` | Schedule block (alpha, sigma) |

---

### `yggdrasil/blocks/positions/`
**Purpose:** Position embeddings.

| File | Purpose |
|------|---------|
| `sinusoidal.py` | Sinusoidal |
| `rope_nd.py` | RoPE n-dimensional |
| `learned.py` | Learned |

---

### `yggdrasil/blocks/data/`
**Purpose:** Dataset blocks for training.

| File | Purpose |
|------|---------|
| `dataset_block.py` | Wraps dataset; outputs batch for loss |

---

### `yggdrasil/blocks/losses/`
**Purpose:** Loss blocks.

| File | Purpose |
|------|---------|
| `loss_block.py` | MSE, L1, etc. |

---

### `yggdrasil/blocks/metrics/`
**Purpose:** Metric blocks (FID, etc.).

| File | Purpose |
|------|---------|
| `metric_block.py` | Evaluation metrics |

---

## `yggdrasil/assemblers/` — Assembly

### `yggdrasil/assemblers/adapter_assembler.py`
**Purpose:** Assembles adapters (ControlNet, LoRA) into graph.

**Role:** Loads weights, wires adapter → backbone.adapter_features.

---

### `yggdrasil/assemblers/model_assembler.py`
**Purpose:** Assembles full model from config.

---

### `yggdrasil/assemblers/pipeline_assembler.py`
**Purpose:** Assembles pipeline (multi-stage) from config.

---

### `yggdrasil/assemblers/multi_modal_assembler.py`
**Purpose:** Assembles multi-modal pipelines.

---

## `yggdrasil/integration/` — External Integrations

### `yggdrasil/integration/diffusers.py`
**Purpose:** Diffusers bridge — convert any diffusers pipeline to ComputeGraph.

**Role:** `DiffusersBridge.from_pretrained()`, `import_pipeline()`. Maps schedulers → YggDrasil solvers, models → backbones. `load_from_diffusers()` returns `ModularDiffusionModel` or `ComputeGraph`.

---

### `yggdrasil/integration/lora_loader.py`
**Purpose:** LoRA loading from HuggingFace / safetensors.

**Role:** `load_lora_weights()`, multi-LoRA merging.

---

## `yggdrasil/training/` — Training

### `yggdrasil/training/graph_trainer.py`
**Purpose:** Train any subset of graph nodes.

**Role:** `train_nodes`, `node_lr`, schedule (freeze/unfreeze by epoch/step), mixed precision, EMA, checkpointing. Works with `ComputeGraph`, not only `ModularDiffusionModel`.

---

### `yggdrasil/training/loss.py`
**Purpose:** Loss functions for diffusion training.

---

### `yggdrasil/training/data.py`
**Purpose:** Data loading utilities.

---

### `yggdrasil/training/trainer.py`
**Purpose:** Legacy trainer (slot-based model).

---

### `yggdrasil/training/checkpoint_ops.py`
**Purpose:** Checkpoint save/load, merge.

---

### `yggdrasil/training/distributed.py`
**Purpose:** Distributed training (DDP, etc.).

---

## `yggdrasil/serving/` — Serving

### `yggdrasil/serving/api.py`
**Purpose:** FastAPI REST API.

**Role:** `POST /generate`, `POST /generate/stream`, `GET /models`, `POST /models/load`, `POST /train/start`, etc. Uses `ModelManager` singleton for loaded models.

---

### `yggdrasil/serving/schema.py`
**Purpose:** Pydantic request/response schemas.

---

### `yggdrasil/serving/gradio_ui.py`
**Purpose:** Gradio UI for generation and training.

---

### `yggdrasil/serving/dynamic_ui.py`
**Purpose:** Dynamic UI generation from plugin schemas.

---

### `yggdrasil/serving/contract_bridge.py`
**Purpose:** Bridges API contract to pipeline execution.

---

### `yggdrasil/serving/middleware.py`
**Purpose:** CORS, auth, etc.

---

### `yggdrasil/serving/queue.py`
**Purpose:** Job queue for async generation.

---

## `yggdrasil/deployment/` — Deployment

### `yggdrasil/deployment/cloud/modal_app.py`
**Purpose:** Modal.com deployment.

---

### `yggdrasil/deployment/cloud/runpod.py`
**Purpose:** RunPod deployment.

---

### `yggdrasil/deployment/cloud/vastai.py`
**Purpose:** Vast.ai deployment.

---

### `yggdrasil/deployment/docker/docker-compose.yaml`
**Purpose:** Docker Compose for local deployment.

---

### `yggdrasil/deployment/export/onnx.py`
**Purpose:** ONNX export for models.

---

## `yggdrasil/plugins/` — Modality Plugins

### `yggdrasil/plugins/base.py`
**Purpose:** `AbstractPlugin` base class.

**Role:** Plugins register blocks, default config, UI schema. `PluginRegistry` holds all plugins.

---

### `yggdrasil/plugins/image/plugin.py`
**Purpose:** Image modality plugin (SD, FLUX, etc.).

---

### `yggdrasil/plugins/audio/plugin.py`
**Purpose:** Audio modality plugin.

---

### `yggdrasil/plugins/video/plugin.py`
**Purpose:** Video modality plugin.

---

### `yggdrasil/plugins/text/plugin.py`
**Purpose:** Text modality plugin.

---

### `yggdrasil/plugins/molecular/plugin.py`
**Purpose:** Molecular diffusion plugin.

---

### `yggdrasil/plugins/threed/plugin.py`
**Purpose:** 3D modality plugin.

---

### `yggdrasil/plugins/timeseries/plugin.py`
**Purpose:** Time-series plugin.

---

### `yggdrasil/plugins/custom/`
**Purpose:** Example custom plugin (backbone, codec, modality).

---

## `yggdrasil/tools/` — Tools

### `yggdrasil/tools/graph_visualizer.py`
**Purpose:** Mermaid/Graphviz diagram generation.

**Role:** `model_to_mermaid()`, `config_to_mermaid()` for model/graph visualization.

---

### `yggdrasil/tools/block_inspector.py`
**Purpose:** Inspect block structure, slots, ports.

---

### `yggdrasil/tools/benchmark.py`
**Purpose:** Benchmark generation speed.

---

## `yggdrasil/ui/` — UI Components

### `yggdrasil/ui/app.py`
**Purpose:** Main UI application assembly.

---

### `yggdrasil/ui/layouts/explore.py`
**Purpose:** Block explorer layout.

---

### `yggdrasil/ui/layouts/generate.py`
**Purpose:** Generation layout.

---

### `yggdrasil/ui/layouts/train.py`
**Purpose:** Training layout.

---

### `yggdrasil/ui/components/block_selector.py`
**Purpose:** Block selection component.

---

### `yggdrasil/ui/components/live_preview.py`
**Purpose:** Live preview component.

---

### `yggdrasil/ui/components/training_panel.py`
**Purpose:** Training controls panel.

---

## `yggdrasil/configs/` — Configuration

### `yggdrasil/configs/__init__.py`
**Purpose:** Load recipes and presets from YAML.

**Role:** `get_recipe(name)`, `list_recipes()`, `get_preset(name)`, `save_preset()`.

---

### `yggdrasil/configs/recipes/`
**Purpose:** Pipeline recipes (model + sampler config).

| File | Purpose |
|------|---------|
| `sd15_generate.yaml` | SD 1.5 text-to-image |
| `sd15_train_lora.yaml` | SD 1.5 LoRA training |
| `flux_generate.yaml` | FLUX generation |
| `audio_generate.yaml` | Audio generation |
| `combined_t2i_img2img.yaml` | T2I + img2img combined |

---

### `yggdrasil/configs/presets/`
**Purpose:** Presets for different model families.

| File | Purpose |
|------|---------|
| `sd15.yaml` | SD 1.5 |
| `sdxl.yaml` | SDXL |
| `flux_dev.yaml` | FLUX |
| `gaussian_3d.yaml` | 3D Gaussian |
| `stable_audio.yaml` | Stable Audio |
| `video_cogvideox.yaml` | CogVideoX |
| etc. | |

---

## `yggdrasil/extensions/` — Extensions

### `yggdrasil/extensions/loader.py`
**Purpose:** Load external extensions (plugins, blocks).

---

## `examples/` — Examples

### `examples/images/README.md`
**Purpose:** Image examples documentation.

---

### `examples/images/compare_diffusers_yggdrasil.py`
**Purpose:** Compare Diffusers vs YggDrasil (SD 1.5, SDXL, SD3).

**Role:** Same prompt/seed/steps; outputs diffusers.png vs yggdrasil.png. `--model sd15|sdxl|sd3`, `--diffusers-only`, `--yggdrasil-only`.

---

### `examples/images/generation_guide.ipynb`
**Purpose:** Jupyter guide for image generation.

---

### `examples/images/sd15/compare_diffusers_yggdrasil.py`
**Purpose:** SD 1.5–specific comparison script.

---

### `examples/images/sd15/YggDrasil_Guide.ipynb`
**Purpose:** SD 1.5 YggDrasil notebook.

---

### `examples/images/sdxl/compare_diffusers_yggdrasil.py`
**Purpose:** SDXL comparison script.

---

### `examples/images/sd3/compare_diffusers_yggdrasil.py`
**Purpose:** SD3 comparison script.

---

## `scripts/` — Scripts

### `scripts/hf_login.py`
**Purpose:** HuggingFace login helper.

---

### `scripts/clear_hf_cache.py`
**Purpose:** Clear HuggingFace cache.

---

## `tests/` — Tests

### `tests/conftest.py`
**Purpose:** Pytest fixtures (device, small models, etc.).

---

### `tests/test_block_system.py`
**Purpose:** Block registry, builder, slots.

---

### `tests/test_lego_constructor.py`
**Purpose:** Lego-style composition tests.

---

### `tests/test_model_families.py`
**Purpose:** SD 1.5, SDXL, SD3, FLUX template tests.

---

### `tests/test_assemblers.py`
**Purpose:** Adapter/model assembler tests.

---

### `tests/test_audio_pipelines.py`
**Purpose:** Audio pipeline tests.

---

### `tests/test_backbones.py`
**Purpose:** Backbone tests.

---

### `tests/test_diffusion_processes.py`
**Purpose:** Diffusion process tests.

---

### `tests/test_plugins.py`
**Purpose:** Plugin registration tests.

---

## Summary

YggDrasil is organized as:

1. **core/** — Block system, ports, slots, ComputeGraph, diffusion process, solvers, model abstractions.
2. **blocks/** — Concrete backbones, codecs, conditioners, adapters, guidances, solvers.
3. **core/graph/** — ComputeGraph, executor, templates, role rules, stages.
4. **pipeline.py** — High-level InferencePipeline / TrainingPipeline API.
5. **integration/diffusers.py** — Bridge from HuggingFace Diffusers.
6. **training/** — GraphTrainer for training any graph nodes.
7. **serving/** — FastAPI, Gradio UI.
8. **plugins/** — Modality-specific plugins (image, audio, video, etc.).
9. **configs/** — YAML recipes and presets.

All components are blocks with `declare_io()` and `process()`; they are composed in a `ComputeGraph` and executed by `GraphExecutor`.
