# yggdrasil/core/graph/orchestrator.py
"""Graph build orchestrator: single point of control for graph assembly.

All node registration, target resolution, deferred bindings, and materialization
go through the orchestrator. No role-based branching in callers — only
RoleRegistry and TargetResolver.

Ref: REFACTORING_GRAPH_PIPELINE_ENGINE.md §3, §10.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phases & constants
# ---------------------------------------------------------------------------

class Phase(str, Enum):
    REGISTER = "REGISTER"
    RESOLVE = "RESOLVE"
    DEFER_OR_CONNECT = "DEFER_OR_CONNECT"
    MATERIALIZE = "MATERIALIZE"
    VALIDATE = "VALIDATE"


DEFER = object()  # Sentinel: target not yet available, defer binding


class DeferKind(str, Enum):
    """How to defer: top-level (loop/codec/ip) or into inner graph (adapter)."""
    TOPLEVEL = "toplevel"   # Add to graph._deferred with target_inner=None
    INNER = "inner"         # Add to graph._deferred with target_inner=loop_name


# ---------------------------------------------------------------------------
# PortNames: standard port names for step graph and graph I/O (§10.D)
# ---------------------------------------------------------------------------

class PortNames:
    """Standard port names to avoid magic strings in templates and registries."""
    # Step (inner loop)
    SAMPLE = "sample"
    TIMESTEP = "timestep"
    CONDITION = "condition"
    UNCOND = "uncond"
    ADAPTER_FEATURES = "adapter_features"
    IMAGE_PROMPT_EMBEDS = "image_prompt_embeds"
    MODEL_OUTPUT = "model_output"
    PREV_SAMPLE = "prev_sample"
    INITIAL_LATENTS = "initial_latents"
    LATENTS = "latents"
    TIMESTEPS = "timesteps"
    # Graph-level
    PROMPT = "prompt"
    NEGATIVE_PROMPT = "negative_prompt"
    CONTROL_IMAGE = "control_image"
    IP_IMAGE = "ip_image"
    NUM_INFERENCE_STEPS = "num_inference_steps"
    GUIDANCE_SCALE = "guidance_scale"
    IMAGE = "image"
    MASK = "mask"
    OUTPUT = "output"
    INPUT = "input"


# ---------------------------------------------------------------------------
# Denoise step contract (§11.4 S2): inputs/outputs of one solver step
# ---------------------------------------------------------------------------

# Standard inputs a step graph may declare (PortNames). Not all are required; depends on backbone/solver.
STEP_INPUT_NAMES = (
    PortNames.LATENTS,
    PortNames.SAMPLE,
    PortNames.TIMESTEP,
    "next_timestep",
    PortNames.CONDITION,
    PortNames.UNCOND,
    PortNames.ADAPTER_FEATURES,
    PortNames.IMAGE_PROMPT_EMBEDS,
)
# Standard outputs: model_output (backbone → solver), prev_sample/next_latents (solver → loop)
STEP_OUTPUT_NAMES = (
    PortNames.MODEL_OUTPUT,
    PortNames.PREV_SAMPLE,
    "next_latents",
)


# ---------------------------------------------------------------------------
# Non-diffusion I/O contract (§11.5 N2): standard keys for segmenter, detector, etc.
# ---------------------------------------------------------------------------

# Standard graph-level input keys (non-diffusion and multi-modal)
NONDIFFUSION_INPUT_KEYS = (
    "image",
    "mask",
    "latents",
    "audio",
    "video",
    "input",
    "prompt",  # text for conditioning
)

# Standard graph-level output keys
NONDIFFUSION_OUTPUT_KEYS = (
    "output",
    "image",
    "mask",
    "latents",
    "audio",
    "video",
    "logits",
    "segmentation_map",
    "depth_map",
    "pose",
    "embeddings",
)


def get_expected_io_for_modality(modality: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return (input_keys, output_keys) typical for the given modality. §11.5 N2.
    Used by executor and UI to know which ports to expose; not enforced."""
    modality = (modality or "").lower()
    if modality == "image":
        return (("image", "mask", "input"), ("output", "image", "segmentation_map", "depth_map", "logits"))
    if modality == "audio":
        return (("audio", "input"), ("output", "audio"))
    if modality == "video":
        return (("video", "image", "input"), ("output", "video"))
    if modality in ("text", "multimodal"):
        return (("prompt", "input", "image", "audio"), ("output", "embeddings"))
    return (NONDIFFUSION_INPUT_KEYS, NONDIFFUSION_OUTPUT_KEYS)


# ---------------------------------------------------------------------------
# NodeRule: rule for a role (single pipeline, no if/elif by role in orchestrator)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NodeRule:
    """Rule describing how to connect a node of a given role.

    Used by TargetResolver to decide (target_graph, target_node, target_port)
    or DEFER. All role-specific logic lives in rules, not in orchestrator code.
    """
    role: str
    target_hint: Optional[str]  # e.g. "backbone", "denoise_loop", or None
    target_port: Optional[str]  # e.g. "adapter_features", "condition"
    output_port: str
    input_port: str
    graph_input_name: Optional[str]  # e.g. "control_image", "prompt"
    insert_into: str  # "root" | "loop_inner"
    defer_if_missing: Tuple[str, ...]  # e.g. ("denoise_loop",) — defer until these exist
    creates_loop: bool  # True for backbone -> substitute with LoopSubGraph


# ---------------------------------------------------------------------------
# BuildState: mutable state during graph build
# ---------------------------------------------------------------------------

@dataclass
class DeferredAdapterBinding:
    """One deferred adapter: to be bound to loop inner graph at materialize."""
    node_name: str
    block_type: str
    config: Dict[str, Any]
    target_inner_node: Optional[str]  # name of loop node in root graph


@dataclass
class BuildState:
    """Orchestrator state during build."""
    graph: Any  # ComputeGraph
    phase: Phase = Phase.REGISTER
    denoise_loop_node_name: Optional[str] = None
    deferred_adapter_bindings: List[DeferredAdapterBinding] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    materialized: bool = False


# ---------------------------------------------------------------------------
# Validation result (structured errors/warnings)
# ---------------------------------------------------------------------------

@dataclass
class ValidationError:
    code: str
    message: str
    node: Optional[str] = None
    port: Optional[str] = None


@dataclass
class ValidationWarning:
    code: str
    message: str
    node: Optional[str] = None
    port: Optional[str] = None


@dataclass
class ValidationResult:
    success: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.success


# ---------------------------------------------------------------------------
# GraphSnapshot: immutable view after materialize (for validation)
# ---------------------------------------------------------------------------

@dataclass
class RegisterResult:
    """Result of register_node: what the graph should do."""
    defer: Optional[DeferKind] = None  # If set, graph should add to _deferred
    target_inner: Optional[str] = None  # For INNER: name of loop node
    connect_now: bool = False           # If True, orchestrator says connect (target in state)
    role: str = "unknown"
    rule: Optional[NodeRule] = None


@dataclass(frozen=True)
class GraphSnapshot:
    topological_order: Tuple[str, ...]
    input_ports: Tuple[str, ...]
    output_ports: Tuple[str, ...]
    loop_node: Optional[str]
    adapter_inputs: Tuple[Tuple[str, str], ...]  # (graph_input_name, node_name), ...


# ---------------------------------------------------------------------------
# AdapterBindingRules: adapter_type -> where to insert, backbone port, graph_input name
# Ref: §5, §11.3 A1
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdapterBindingRule:
    """Where and how to bind an adapter type into the graph."""
    adapter_type: str
    insert_into: str  # "loop_inner" | "root"
    backbone_port: str  # "adapter_features" | "image_prompt_embeds"
    graph_input_name: str  # "control_image" | "ip_image" | "control_image_<node_name>"
    input_port: str  # block input port name


def _default_adapter_bindings() -> Dict[str, AdapterBindingRule]:
    return {
        "adapter/controlnet": AdapterBindingRule(
            adapter_type="adapter/controlnet",
            insert_into="loop_inner",
            backbone_port="adapter_features",
            graph_input_name="control_image",
            input_port="control_image",
        ),
        "adapter/t2i_adapter": AdapterBindingRule(
            adapter_type="adapter/t2i_adapter",
            insert_into="loop_inner",
            backbone_port="adapter_features",
            graph_input_name="control_image",
            input_port="control_image",
        ),
        "adapter/ip_adapter": AdapterBindingRule(
            adapter_type="adapter/ip_adapter",
            insert_into="root",
            backbone_port="image_prompt_embeds",
            graph_input_name="ip_image",
            input_port="image_features",
        ),
        # Motion adapter (AnimateDiff) is typically part of backbone; no separate graph_input
        "adapter/motion": AdapterBindingRule(
            adapter_type="adapter/motion",
            insert_into="loop_inner",
            backbone_port="motion_features",
            graph_input_name="motion_image",
            input_port="motion_image",
        ),
    }


class AdapterBindingRules:
    """Registry: adapter_type -> AdapterBindingRule. Extensible without changing orchestrator."""

    def __init__(self) -> None:
        self._rules: Dict[str, AdapterBindingRule] = _default_adapter_bindings()

    def register(self, adapter_type: str, rule: AdapterBindingRule) -> None:
        self._rules[adapter_type] = rule

    def get(self, adapter_type: str) -> Optional[AdapterBindingRule]:
        return self._rules.get(adapter_type)

    def get_or_prefix(self, adapter_type: str) -> Optional[AdapterBindingRule]:
        """Return rule for exact type or first matching prefix (e.g. adapter/controlnet_*)."""
        if adapter_type in self._rules:
            return self._rules[adapter_type]
        for prefix, rule in sorted(self._rules.items(), key=lambda x: -len(x[0])):
            if adapter_type.startswith(prefix):
                return rule
        return None


# Optional graph inputs: set to None when not provided (avoid sticky cache, §5.3, §11.3 A3)
OPTIONAL_GRAPH_INPUT_PATTERNS: Tuple[str, ...] = (
    "ip_image",
    "control_image",
    "t2i_control_image",
    "source_image",
)


# ---------------------------------------------------------------------------
# LoopTemplates: (solver_type, modality) -> step template (§11.2 L2)
# ---------------------------------------------------------------------------

class LoopTemplates:
    """Registry for denoise loop step templates. Extensible without changing orchestrator. §11.2 L2, L3.

    Uses class-level storage so that register() persists across all LoopTemplates() instances
    (e.g. get_step_template_id_for_metadata() will see custom (solver_type, modality) -> template_id).
    """

    DEFAULT_GENERIC = "generic"

    _templates: Dict[Tuple[str, str], Any] = {}  # class-level, shared
    _default: Optional[str] = DEFAULT_GENERIC

    def __init__(self) -> None:
        pass

    def register(self, solver_type: str, modality: str, template: Any) -> None:
        """Register a step template for (solver_type, modality). template can be id or callable."""
        LoopTemplates._templates[(solver_type, modality)] = template

    def get(self, solver_type: str, modality: str = "image") -> Optional[Any]:
        """Return template for (solver_type, modality) or None (caller may use get_or_default)."""
        return (
            LoopTemplates._templates.get((solver_type, modality))
            or LoopTemplates._templates.get((solver_type, "*"))
        )

    def get_or_default(self, solver_type: str, modality: str = "image") -> Any:
        """Return template for (solver_type, modality) or default (generic). §11.2 L3."""
        return self.get(solver_type, modality) or LoopTemplates._default or self.DEFAULT_GENERIC

    def set_default(self, template_id: str) -> None:
        """Set default template id when no specific match (e.g. 'generic')."""
        LoopTemplates._default = template_id


def get_step_template_id_for_metadata(metadata: Optional[Dict[str, Any]] = None) -> str:
    """Return step template id for given graph/loop metadata. §11.2 L1 (задел для подмены backbone на цикл).

    If metadata.base_model is sdxl/sd15, returns "step_sdxl" (UNet + batched CFG). Otherwise
    uses LoopTemplates.get_or_default(solver_type, modality).
    """
    meta = metadata or {}
    base = meta.get("base_model") or ""
    if base in ("sdxl", "sd15"):
        return "step_sdxl"
    if base == "sd3":
        return "step_sd3"
    if base == "flux":
        return "step_flux"
    solver_type = meta.get("solver_type") or "euler_discrete"
    modality = meta.get("modality") or "image"
    return LoopTemplates().get_or_default(solver_type, modality)


# ---------------------------------------------------------------------------
# StepBuilderRegistry: template_id -> step graph builder (§11 L1 extension)
# ---------------------------------------------------------------------------

class StepBuilderRegistry:
    """Registry of step graph builders by template_id. L1 extension for SD3, Flux, etc.

    Register a callable(metadata, pretrained, num_steps, guidance_scale, **kwargs) -> ComputeGraph
    for a template_id (e.g. "step_sdxl", "step_flux", "generic"). When building a denoise loop,
    get_step_template_id_for_metadata() yields template_id; if a builder is registered
    it is used to build the inner step graph, else the default builder is used.
    Uses a class-level dict so that registrations (e.g. "generic" from image_pipelines)
    are shared across all StepBuilderRegistry() instances.
    """

    _builders: Dict[str, Callable[..., Any]] = {}  # class-level, shared

    def __init__(self) -> None:
        pass

    def register(self, template_id: str, builder: Callable[..., Any]) -> None:
        """Register a step graph builder for template_id."""
        StepBuilderRegistry._builders[template_id] = builder

    def get(self, template_id: str) -> Optional[Callable[..., Any]]:
        """Return builder for template_id or None."""
        return StepBuilderRegistry._builders.get(template_id)

    def get_or_default(
        self,
        template_id: str,
        default: Optional[Callable[..., Any]] = None,
    ) -> Optional[Callable[..., Any]]:
        """Return builder for template_id or default."""
        return self.get(template_id) or default


# ---------------------------------------------------------------------------
# SolverRegistry: Diffusers scheduler -> solver_type (§S1, L5)
# ---------------------------------------------------------------------------

class SolverRegistry:
    """Maps Diffusers scheduler class names / ids to internal solver_type for step template lookup.

    Uses class-level _map so that registrations are shared across all instances (e.g. Diffusers
    integration and custom register() see the same mapping).
    """

    _map: Dict[str, Tuple[str, Optional[str]]] = {}
    _defaults_registered: bool = False

    def __init__(self) -> None:
        if not SolverRegistry._defaults_registered:
            SolverRegistry._register_class_defaults()
            SolverRegistry._defaults_registered = True

    @classmethod
    def _register_class_defaults(cls) -> None:
        """Default mapping from Diffusers scheduler names to solver_type. Ref: diffusers schedulers overview."""
        defaults = [
            ("EulerDiscreteScheduler", "euler_discrete", "epsilon"),
            ("EulerAncestralDiscreteScheduler", "euler_ancestral_discrete", "epsilon"),
            ("DDIMScheduler", "ddim", "epsilon"),
            ("DDPMScheduler", "ddpm", "epsilon"),
            ("DPMSolverMultistepScheduler", "dpm_multistep", "epsilon"),
            ("DPMSolverSinglestepScheduler", "dpm_singlestep", "epsilon"),
            ("LCMScheduler", "lcm", "epsilon"),
            ("FlowMatchEulerDiscreteScheduler", "flow_match_euler", "v_prediction"),
            ("UniPCMultistepScheduler", "unipc_multistep", "epsilon"),
            ("DEISMultistepScheduler", "deis_multistep", "epsilon"),
            ("HeunDiscreteScheduler", "heun_discrete", "epsilon"),
            ("LMSDiscreteScheduler", "lms_discrete", "epsilon"),
            ("euler_discrete", "euler_discrete", None),
            ("ddim", "ddim", None),
            ("lcm", "lcm", None),
        ]
        for sid, solver_type, sig in defaults:
            cls._map[sid] = (solver_type, sig)

    def register(self, scheduler_id: str, solver_type: str, step_signature: Optional[str] = None) -> None:
        """Register scheduler_id (e.g. 'EulerDiscreteScheduler') -> (solver_type, step_signature)."""
        SolverRegistry._map[scheduler_id] = (solver_type, step_signature)

    def get_solver_type(self, scheduler_id: str) -> Optional[str]:
        """Return internal solver_type for Diffusers scheduler id, or None."""
        entry = SolverRegistry._map.get(scheduler_id)
        return entry[0] if entry else None

    def get_step_signature(self, scheduler_id: str) -> Optional[str]:
        """Return step signature hint (e.g. 'epsilon', 'v_prediction') for scheduler, or None."""
        entry = SolverRegistry._map.get(scheduler_id)
        return entry[1] if entry else None


def graph_to_mermaid(graph: Any) -> str:
    """Export graph to Mermaid diagram string (§10.I). Uses graph.visualize() if available."""
    if hasattr(graph, "visualize") and callable(graph.visualize):
        return graph.visualize()
    return "graph LR\n    empty[empty]"


def get_optional_graph_input_names(graph: Any) -> List[str]:
    """Return graph input names that are optional (control/ip etc.). Caller may set to None when not in kwargs."""
    names = list(getattr(graph, "graph_inputs", {}).keys())
    out: List[str] = []
    for n in names:
        if n in OPTIONAL_GRAPH_INPUT_PATTERNS:
            out.append(n)
        if n.startswith("control_image_"):
            out.append(n)
    return out


# ---------------------------------------------------------------------------
# RoleRegistry: block_type / role -> NodeRule
# ---------------------------------------------------------------------------

def _default_rules() -> Dict[str, NodeRule]:
    """Build default role -> NodeRule from existing role_rules semantics."""
    return {
        "adapter": NodeRule(
            role="adapter",
            target_hint="backbone",
            target_port="adapter_features",
            output_port="output",
            input_port="control_image",
            graph_input_name="control_image",
            insert_into="loop_inner",
            defer_if_missing=("denoise_loop",),
            creates_loop=False,
        ),
        "inner_module": NodeRule(
            role="inner_module",
            target_hint="backbone",
            target_port="adapter_features",
            output_port="adapter_features",
            input_port="control_image",
            graph_input_name="control_image",
            insert_into="loop_inner",
            defer_if_missing=("denoise_loop",),
            creates_loop=False,
        ),
        "conditioner": NodeRule(
            role="conditioner",
            target_hint=None,
            target_port=None,
            output_port="embedding",
            input_port="text",
            graph_input_name=None,
            insert_into="root",
            defer_if_missing=(),
            creates_loop=False,
        ),
        "processor": NodeRule(
            role="processor",
            target_hint=None,
            target_port=None,
            output_port="output",
            input_port="input",
            graph_input_name="input",
            insert_into="root",
            defer_if_missing=(),
            creates_loop=False,
        ),
        "outer_module": NodeRule(
            role="outer_module",
            target_hint=None,
            target_port=None,
            output_port="output",
            input_port="input",
            graph_input_name="input",
            insert_into="root",
            defer_if_missing=(),
            creates_loop=False,
        ),
        "backbone": NodeRule(
            role="backbone",
            target_hint="denoise_loop",
            target_port=None,
            output_port="output",
            input_port="input",
            graph_input_name=None,
            insert_into="root",
            defer_if_missing=(),
            creates_loop=True,
        ),
        "codec": NodeRule(
            role="codec",
            target_hint="denoise_loop",
            target_port="latents",
            output_port="decoded",
            input_port="latent",
            graph_input_name=None,
            insert_into="root",
            defer_if_missing=("denoise_loop",),
            creates_loop=False,
        ),
        "solver": NodeRule(
            role="solver",
            target_hint=None,
            target_port=None,
            output_port="prev_sample",
            input_port="input",
            graph_input_name=None,
            insert_into="loop_inner",
            defer_if_missing=(),
            creates_loop=False,
        ),
        "denoise_loop": NodeRule(
            role="denoise_loop",
            target_hint=None,
            target_port=None,
            output_port="latents",
            input_port="initial_latents",
            graph_input_name=None,
            insert_into="root",
            defer_if_missing=(),
            creates_loop=False,
        ),
        # Non-diffusion roles (§7): root-only, no loop, no defer
        "segmenter": NodeRule(
            role="segmenter",
            target_hint=None,
            target_port=None,
            output_port="output",
            input_port="image",
            graph_input_name="image",
            insert_into="root",
            defer_if_missing=(),
            creates_loop=False,
        ),
        "detector": NodeRule(
            role="detector",
            target_hint=None,
            target_port=None,
            output_port="output",
            input_port="image",
            graph_input_name="image",
            insert_into="root",
            defer_if_missing=(),
            creates_loop=False,
        ),
        "classifier": NodeRule(
            role="classifier",
            target_hint=None,
            target_port=None,
            output_port="output",
            input_port="input",
            graph_input_name="input",
            insert_into="root",
            defer_if_missing=(),
            creates_loop=False,
        ),
        "depth_estimator": NodeRule(
            role="depth_estimator",
            target_hint=None,
            target_port=None,
            output_port="output",
            input_port="image",
            graph_input_name="image",
            insert_into="root",
            defer_if_missing=(),
            creates_loop=False,
        ),
        "pose_estimator": NodeRule(
            role="pose_estimator",
            target_hint=None,
            target_port=None,
            output_port="output",
            input_port="image",
            graph_input_name="image",
            insert_into="root",
            defer_if_missing=(),
            creates_loop=False,
        ),
        "super_resolution": NodeRule(
            role="super_resolution",
            target_hint=None,
            target_port=None,
            output_port="output",
            input_port="input",
            graph_input_name="input",
            insert_into="root",
            defer_if_missing=(),
            creates_loop=False,
        ),
        "style_encoder": NodeRule(
            role="style_encoder",
            target_hint=None,
            target_port=None,
            output_port="output",
            input_port="input",
            graph_input_name="input",
            insert_into="root",
            defer_if_missing=(),
            creates_loop=False,
        ),
        "feature_extractor": NodeRule(
            role="feature_extractor",
            target_hint=None,
            target_port=None,
            output_port="output",
            input_port="input",
            graph_input_name="input",
            insert_into="root",
            defer_if_missing=(),
            creates_loop=False,
        ),
        # T3: loss node — output "loss" used for backward in TrainingPipeline
        "loss": NodeRule(
            role="loss",
            target_hint=None,
            target_port=None,
            output_port="loss",
            input_port="prediction",
            graph_input_name=None,
            insert_into="root",
            defer_if_missing=(),
            creates_loop=False,
        ),
    }


# Custom roles/plugins: merged into default registry (§11.4 S4)
_custom_type_to_role: Dict[str, str] = {}
_custom_rules: Dict[str, NodeRule] = {}


def register_custom_role(block_type_prefix: str, rule: NodeRule) -> None:
    """Register a custom role for a block_type prefix (plugins, world models). No change to core code.

    Example: register_custom_role("world_model/", NodeRule(role="world_model", ...))
    """
    _custom_type_to_role[block_type_prefix] = rule.role
    _custom_rules[rule.role] = rule


class RoleRegistry:
    """Maps block_type / role -> NodeRule. Extensible without changing orchestrator."""

    def __init__(self) -> None:
        self._rules: Dict[str, NodeRule] = {**_default_rules(), **_custom_rules}
        self._type_to_role: Dict[str, str] = dict(_custom_type_to_role)

    def register_type_to_role(self, block_type_prefix: str, role: str) -> None:
        """Register block_type prefix -> role (e.g. 'adapter/' -> 'adapter')."""
        self._type_to_role[block_type_prefix] = role

    def register_rule(self, role: str, rule: NodeRule) -> None:
        """Register or override rule for a role."""
        self._rules[role] = rule

    def get_role_for_block_type(self, block_type: str) -> str:
        """Resolve role from block_type (longest prefix match). Custom roles first, then built-in."""
        if not block_type:
            return "unknown"
        combined = {**_custom_type_to_role, **self._type_to_role}
        for prefix, role in sorted(combined.items(), key=lambda x: -len(x[0])):
            if block_type.startswith(prefix):
                return role
        # Fallback: use built-in TYPE_TO_ROLE from role_rules
        from .role_rules import TYPE_TO_ROLE
        for prefix, role in sorted(TYPE_TO_ROLE.items(), key=lambda x: -len(x[0])):
            if block_type.startswith(prefix):
                return role
        return "unknown"

    def get_rule(self, role: str) -> Optional[NodeRule]:
        """Return NodeRule for role, or None."""
        return self._rules.get(role)


# ---------------------------------------------------------------------------
# TargetResolver: (NodeRule, BuildState) -> target or DEFER
# ---------------------------------------------------------------------------

def resolve_target(rule: NodeRule, state: BuildState) -> Any:
    """Resolve where to connect a node: (graph, node_name, port) or DEFER.

    Returns:
        Tuple[(graph, node_name, port)] for immediate connect, or DEFER.
    """
    if rule.creates_loop:
        # Backbone: we don't connect to existing node; we substitute with loop.
        return DEFER  # Handled by caller: add as loop/codec deferred

    for missing in rule.defer_if_missing:
        if missing == "denoise_loop" and state.denoise_loop_node_name is None:
            return DEFER
        # Could extend for other symbols

    if rule.insert_into == "loop_inner" and state.denoise_loop_node_name:
        graph = state.graph
        loop_name = state.denoise_loop_node_name
        loop_block = graph.nodes.get(loop_name)
        if loop_block is None:
            return DEFER
        inner = getattr(loop_block, "graph", None) or (
            getattr(loop_block, "_loop", None) and getattr(loop_block._loop, "graph", None)
        )
        if inner is None:
            return DEFER
        target_node = rule.target_hint or "backbone"
        if target_node not in inner.nodes:
            for n, b in inner.nodes.items():
                if getattr(b, "block_type", "").startswith("backbone/"):
                    target_node = n
                    break
        if target_node in inner.nodes and rule.target_port:
            return (inner, target_node, rule.target_port)
    elif rule.insert_into == "root" and rule.target_hint and rule.target_port:
        if rule.target_hint == "denoise_loop" and state.denoise_loop_node_name:
            target_name = state.denoise_loop_node_name
            return (state.graph, target_name, rule.target_port)
    return DEFER


# ---------------------------------------------------------------------------
# GraphBuildOrchestrator
# ---------------------------------------------------------------------------

class GraphBuildOrchestrator:
    """Single point of control for graph assembly: phases REGISTER -> RESOLVE -> DEFER_OR_CONNECT -> MATERIALIZE -> VALIDATE."""

    def __init__(self, graph: Any, *, role_registry: Optional[RoleRegistry] = None) -> None:
        self.graph = graph
        self.registry = role_registry or RoleRegistry()
        self._state = BuildState(graph=graph, metadata=dict(getattr(graph, "metadata", None) or {}))
        self._snapshot: Optional[GraphSnapshot] = None
        self._on_phase: Optional[Any] = None  # Callable[[Phase, BuildState], None]

    def get_state(self) -> BuildState:
        return self._state

    def set_metadata(self, key: str, value: Any) -> None:
        self._state.metadata[key] = value

    def get_metadata(self, key: str) -> Any:
        return self._state.metadata.get(key)

    def set_on_phase(self, callback: Optional[Any]) -> None:
        """Set optional callback(phase, state) after each phase (debugging)."""
        self._on_phase = callback

    def _emit_phase(self, phase: Phase) -> None:
        self._state.phase = phase
        if self._on_phase:
            try:
                self._on_phase(phase, self._state)
            except Exception as e:
                logger.debug("on_phase callback failed: %s", e)

    def register_node(
        self,
        name: str,
        block_type: str,
        config: Dict[str, Any],
        *,
        target_inner: Optional[str] = None,
    ) -> RegisterResult:
        """Run REGISTER -> RESOLVE -> DEFER_OR_CONNECT for one node.

        Returns RegisterResult so the graph knows: defer toplevel, defer inner, or connect now.
        """
        self._emit_phase(Phase.REGISTER)
        role = self.registry.get_role_for_block_type(block_type)
        rule = self.registry.get_rule(role)
        result = RegisterResult(role=role, rule=rule)

        self._emit_phase(Phase.RESOLVE)
        target = resolve_target(rule, self._state) if rule else DEFER
        if target_inner is not None:
            target = (None, target_inner, None)

        self._emit_phase(Phase.DEFER_OR_CONNECT)
        if rule and rule.creates_loop:
            # Backbone -> loop: graph will substitute with loop and defer; we just record loop name
            self._state.denoise_loop_node_name = name
            result.defer = DeferKind.TOPLEVEL
            return result
        if block_type.startswith("loop/"):
            self._state.denoise_loop_node_name = name
            result.defer = DeferKind.TOPLEVEL
            return result
        if block_type.startswith("codec/") or block_type == "adapter/ip_adapter":
            result.defer = DeferKind.TOPLEVEL
            return result
        if target is DEFER or (isinstance(target, tuple) and len(target) == 3 and target[0] is None and target_inner):
            # Adapter or inner: defer to loop inner
            inner_name = target_inner or self._state.denoise_loop_node_name
            self._state.deferred_adapter_bindings.append(
                DeferredAdapterBinding(
                    node_name=name,
                    block_type=block_type,
                    config=config,
                    target_inner_node=inner_name,
                )
            )
            result.defer = DeferKind.INNER
            result.target_inner = inner_name
            return result
        if isinstance(target, tuple) and len(target) == 3 and target[0] is not None:
            result.connect_now = True
            return result
        return result

    def update_denoise_loop_name(self, name: str) -> None:
        """Set the name of the denoise loop node (e.g. after adding placeholder)."""
        self._state.denoise_loop_node_name = name

    def materialize(self) -> None:
        """Run MATERIALIZE: build deferred blocks, wire adapters, then validate."""
        if self._state.materialized:
            return
        self._emit_phase(Phase.MATERIALIZE)
        # Explicit error when adapters were deferred but no loop exists (§10.G)
        deferred = self._state.deferred_adapter_bindings
        loop_name = self._state.denoise_loop_node_name
        if deferred:
            if not loop_name or loop_name not in getattr(self.graph, "nodes", {}):
                names = [b.node_name for b in deferred]
                raise ValueError(
                    "Cannot materialize: the following adapter(s) were deferred for binding to a denoise loop, "
                    "but no loop node exists in the graph. Add a backbone or loop first, or remove the adapter(s). "
                    f"Deferred: {names}"
                )
        # Delegate to graph's existing materialization (build blocks, wire)
        if getattr(self.graph, "_deferred", None):
            self.graph._materialize_deferred()
        self._state.materialized = True
        self._state.deferred_adapter_bindings.clear()
        # Re-detect loop node after materialize
        for n, block in self.graph.nodes.items():
            bt = getattr(block, "block_type", "")
            if bt.startswith("loop/"):
                self._state.denoise_loop_node_name = n
                break
        self._snapshot = self._build_snapshot()
        self._emit_phase(Phase.VALIDATE)

    def _build_snapshot(self) -> Optional[GraphSnapshot]:
        """Build immutable snapshot after materialize."""
        try:
            order = tuple(self.graph.nodes.keys())
            inputs = tuple(self.graph.graph_inputs.keys())
            outputs = tuple(self.graph.graph_outputs.keys())
            loop_node = self._state.denoise_loop_node_name
            adapter_inputs: List[Tuple[str, str]] = []
            for inp in inputs:
                if inp.startswith("control_image") or inp == "ip_image":
                    targets = self.graph.graph_inputs.get(inp, [])
                    if targets:
                        adapter_inputs.append((inp, targets[0][0]))
            return GraphSnapshot(
                topological_order=order,
                input_ports=inputs,
                output_ports=outputs,
                loop_node=loop_node,
                adapter_inputs=tuple(adapter_inputs),
            )
        except Exception:
            return None

    def get_snapshot(self) -> Optional[GraphSnapshot]:
        """Return snapshot after materialize, or None."""
        return self._snapshot

    def validate(self) -> ValidationResult:
        """Run VALIDATE; return structured errors and warnings."""
        self._emit_phase(Phase.VALIDATE)
        errors: List[ValidationError] = []
        warnings: List[ValidationWarning] = []

        # Required inputs that must have a source (graph has them declared)
        for inp_name, targets in getattr(self.graph, "graph_inputs", {}).items():
            if not targets:
                errors.append(ValidationError("MISSING_TARGET", f"Graph input '{inp_name}' has no target", node=None, port=inp_name))

        # Duplicate adapter input names
        seen: Set[str] = set()
        for inp in getattr(self.graph, "graph_inputs", {}).keys():
            if inp in seen:
                warnings.append(ValidationWarning("DUPLICATE_INPUT", f"Graph input '{inp}' declared more than once", port=inp))
            seen.add(inp)

        return ValidationResult(
            success=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def invalidate(self) -> None:
        """Mark graph dirty after add_node/replace_node so next materialize runs."""
        self._state.materialized = False
        self._snapshot = None
