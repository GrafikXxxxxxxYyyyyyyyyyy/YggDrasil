"""YggDrasil Compute Graph — настоящий dataflow Lego-конструктор.

ComputeGraph — DAG из блоков, соединённых через типизированные порты.
Это главная структура данных для сборки произвольных диффузионных pipeline.
"""

from .graph import ComputeGraph, Edge
from .executor import GraphExecutor
from .model_graph_builder import build_model_graph
from .subgraph import SubGraph
from .stage import AbstractStage
from .adapters import (
    add_controlnet_to_graph,
    add_ip_adapter_to_graph,
    add_adapter_to_graph,
    get_controlnet_input_mapping,
    get_cross_attention_dim_from_graph,
)
from .orchestrator import (
    GraphBuildOrchestrator,
    BuildState,
    NodeRule,
    RoleRegistry,
    Phase,
    ValidationResult,
    ValidationError,
    ValidationWarning,
    GraphSnapshot,
    RegisterResult,
    AdapterBindingRule,
    AdapterBindingRules,
    get_optional_graph_input_names,
    PortNames,
    STEP_INPUT_NAMES,
    STEP_OUTPUT_NAMES,
    NONDIFFUSION_INPUT_KEYS,
    NONDIFFUSION_OUTPUT_KEYS,
    get_expected_io_for_modality,
    LoopTemplates,
    StepBuilderRegistry,
    SolverRegistry,
    register_custom_role,
    graph_to_mermaid,
    get_step_template_id_for_metadata,
)

__all__ = [
    "AbstractStage",
    "AdapterBindingRule",
    "AdapterBindingRules",
    "BuildState",
    "ComputeGraph",
    "Edge",
    "GraphBuildOrchestrator",
    "GraphSnapshot",
    "NodeRule",
    "Phase",
    "PortNames",
    "STEP_INPUT_NAMES",
    "STEP_OUTPUT_NAMES",
    "NONDIFFUSION_INPUT_KEYS",
    "NONDIFFUSION_OUTPUT_KEYS",
    "get_expected_io_for_modality",
    "RegisterResult",
    "register_custom_role",
    "RoleRegistry",
    "ValidationError",
    "ValidationResult",
    "ValidationWarning",
    "get_optional_graph_input_names",
    "graph_to_mermaid",
    "get_step_template_id_for_metadata",
    "GraphExecutor",
    "LoopTemplates",
    "StepBuilderRegistry",
    "SolverRegistry",
    "SubGraph",
    "add_controlnet_to_graph",
    "add_ip_adapter_to_graph",
    "add_adapter_to_graph",
    "get_controlnet_input_mapping",
    "get_cross_attention_dim_from_graph",
    "build_model_graph",
]
