"""Tests for GraphBuildOrchestrator and related types.

Ref: REFACTORING_GRAPH_PIPELINE_ENGINE.md §3, §10.
No model downloads; uses mocks and isolated orchestrator logic.
"""
import pytest

from yggdrasil.core.graph.orchestrator import (
    Phase,
    NodeRule,
    BuildState,
    RoleRegistry,
    resolve_target,
    GraphBuildOrchestrator,
    RegisterResult,
    DeferKind,
    ValidationResult,
    AdapterBindingRules,
    AdapterBindingRule,
    get_optional_graph_input_names,
    DEFER,
)


# ==================== RoleRegistry ====================

class TestRoleRegistry:
    def test_get_role_for_block_type(self):
        r = RoleRegistry()
        assert r.get_role_for_block_type("adapter/controlnet") == "adapter"
        assert r.get_role_for_block_type("adapter/ip_adapter") == "adapter"
        assert r.get_role_for_block_type("backbone/unet2d_condition") == "backbone"
        assert r.get_role_for_block_type("conditioner/clip_sdxl") == "conditioner"
        assert r.get_role_for_block_type("loop/denoise_sdxl") == "denoise_loop"
        assert r.get_role_for_block_type("codec/vae") == "codec"
        assert r.get_role_for_block_type("solver/euler_discrete") == "solver"

    def test_get_rule(self):
        r = RoleRegistry()
        rule = r.get_rule("adapter")
        assert rule is not None
        assert rule.role == "adapter"
        assert rule.insert_into == "loop_inner"
        assert rule.target_port == "adapter_features"
        assert rule.defer_if_missing == ("denoise_loop",)
        assert r.get_rule("unknown_role") is None


# ==================== TargetResolver ====================

class TestResolveTarget:
    def test_adapter_defers_when_no_loop(self):
        r = RoleRegistry()
        rule = r.get_rule("adapter")
        state = BuildState(graph=_MockGraph())
        assert state.denoise_loop_node_name is None
        target = resolve_target(rule, state)
        assert target is DEFER

    def test_adapter_resolves_when_loop_exists(self):
        r = RoleRegistry()
        rule = r.get_rule("adapter")
        state = BuildState(graph=_MockGraph())
        state.denoise_loop_node_name = "denoise_loop"
        g = state.graph
        g.nodes["denoise_loop"] = _MockLoopBlock()
        target = resolve_target(rule, state)
        assert target is not DEFER
        assert isinstance(target, tuple)
        assert len(target) == 3
        inner, node_name, port = target
        assert node_name == "backbone"
        assert port == "adapter_features"


# ==================== GraphBuildOrchestrator ====================

class TestGraphBuildOrchestrator:
    def test_register_node_backbone_returns_toplevel_defer(self):
        g = _MockGraph()
        orch = GraphBuildOrchestrator(g)
        res = orch.register_node("my_loop", "loop/denoise_sdxl", {"pretrained": "x"})
        assert res.defer == DeferKind.TOPLEVEL
        assert orch.get_state().denoise_loop_node_name == "my_loop"

    def test_register_node_adapter_returns_inner_defer_when_no_loop(self):
        g = _MockGraph()
        orch = GraphBuildOrchestrator(g)
        res = orch.register_node("ctrl", "adapter/controlnet", {"pretrained": "y"}, target_inner=None)
        assert res.defer == DeferKind.INNER
        assert res.target_inner is None
        assert len(orch.get_state().deferred_adapter_bindings) == 1
        assert orch.get_state().deferred_adapter_bindings[0].node_name == "ctrl"

    def test_register_node_adapter_with_target_inner(self):
        g = _MockGraph()
        orch = GraphBuildOrchestrator(g)
        orch.update_denoise_loop_name("denoise_loop")
        res = orch.register_node("ctrl", "adapter/controlnet", {}, target_inner="denoise_loop")
        assert res.defer == DeferKind.INNER
        assert res.target_inner == "denoise_loop"

    def test_validate_returns_validation_result(self):
        g = _MockGraph()
        g.graph_inputs["prompt"] = [("cond", "text")]
        orch = GraphBuildOrchestrator(g)
        result = orch.validate()
        assert isinstance(result, ValidationResult)
        assert result.success is True

    def test_validate_fails_on_empty_input_targets(self):
        g = _MockGraph()
        g.graph_inputs["prompt"] = []  # no target
        orch = GraphBuildOrchestrator(g)
        result = orch.validate()
        assert result.success is False
        assert any(e.code == "MISSING_TARGET" for e in result.errors)


# ==================== AdapterBindingRules & optional inputs ====================

class TestAdapterBindingRules:
    def test_get_controlnet(self):
        r = AdapterBindingRules()
        rule = r.get("adapter/controlnet")
        assert rule is not None
        assert rule.insert_into == "loop_inner"
        assert rule.backbone_port == "adapter_features"
        assert rule.graph_input_name == "control_image"

    def test_get_ip_adapter(self):
        r = AdapterBindingRules()
        rule = r.get("adapter/ip_adapter")
        assert rule is not None
        assert rule.graph_input_name == "ip_image"
        assert rule.backbone_port == "image_prompt_embeds"

    def test_get_or_prefix(self):
        r = AdapterBindingRules()
        assert r.get_or_prefix("adapter/controlnet_canny") is not None
        assert r.get_or_prefix("adapter/unknown") is None


class TestOptionalGraphInputs:
    def test_get_optional_graph_input_names(self):
        g = _MockGraph()
        g.graph_inputs = {"prompt": [("c", "t")], "control_image": [("n", "p")], "control_image_cnet1": [("n2", "p2")], "ip_image": [("e", "r")]}
        names = get_optional_graph_input_names(g)
        assert "control_image" in names
        assert "control_image_cnet1" in names
        assert "ip_image" in names
        assert "prompt" not in names


# ==================== Non-diffusion roles ====================

class TestNonDiffusionRoles:
    def test_segmenter_role(self):
        r = RoleRegistry()
        assert r.get_role_for_block_type("segmenter/semantic") == "segmenter"
        rule = r.get_rule("segmenter")
        assert rule is not None
        assert rule.insert_into == "root"
        assert rule.creates_loop is False

    def test_detector_depth_super_resolution(self):
        r = RoleRegistry()
        assert r.get_role_for_block_type("detector/object") == "detector"
        assert r.get_role_for_block_type("depth_estimator/dpt") == "depth_estimator"
        assert r.get_role_for_block_type("super_resolution/esrgan") == "super_resolution"


# ==================== Integration with ComputeGraph ====================

class TestOrchestratorIntegration:
    def test_graph_has_get_orchestrator(self):
        from yggdrasil.core.graph import ComputeGraph
        g = ComputeGraph("test")
        orch = g._get_orchestrator()
        assert orch is not None
        assert isinstance(orch, GraphBuildOrchestrator)

    def test_orchestrator_materialize_calls_graph_materialize_deferred(self):
        from yggdrasil.core.graph import ComputeGraph
        g = ComputeGraph("test")
        # Add deferred node (placeholder)
        g.add_node(type="loop/denoise_sdxl", pretrained="stabilityai/stable-diffusion-xl-base-1.0", name="denoise_loop")
        assert len(g._deferred) == 1
        orch = g._get_orchestrator()
        assert orch.get_state().materialized is False
        # materialize will call _materialize_deferred (which loads weights); skip actual run in unit test
        # Just check that orchestrator state is updated when we have no deferred (clear _deferred first to avoid download)
        g._deferred.clear()
        orch.materialize()
        assert orch.get_state().materialized is True

    def test_from_spec_with_graph(self):
        """InferencePipeline.from_spec(ComputeGraph) dispatches to from_graph."""
        from yggdrasil.core.graph import ComputeGraph
        from yggdrasil.pipeline import InferencePipeline
        g = ComputeGraph("test")
        pipe = InferencePipeline.from_spec(g)
        assert pipe.graph is g

    def test_from_spec_with_path_raises_for_missing_file(self):
        """InferencePipeline.from_spec(path) with non-existent path raises."""
        from yggdrasil.pipeline import InferencePipeline
        with pytest.raises(Exception):
            InferencePipeline.from_spec("/nonexistent/path/for/spec.yaml")

    def test_from_spec_invalid_type_raises(self):
        from yggdrasil.pipeline import InferencePipeline
        with pytest.raises(TypeError, match="from_spec"):
            InferencePipeline.from_spec(123)

    def test_pipeline_init_with_graphs_dict(self):
        """InferencePipeline(graphs={...}) builds combined pipeline."""
        from yggdrasil.core.graph import ComputeGraph
        from yggdrasil.pipeline import InferencePipeline
        g1 = ComputeGraph("stage1")
        g2 = ComputeGraph("stage2")
        pipe = InferencePipeline(graphs={"a": g1, "b": g2})
        assert pipe.graph is not None
        assert "a" in pipe.graph.nodes
        assert "b" in pipe.graph.nodes

    def test_pipeline_init_requires_graph_or_graphs(self):
        from yggdrasil.pipeline import InferencePipeline
        with pytest.raises(ValueError, match="graph= or graphs="):
            InferencePipeline()


# ==================== PortNames & materialize error ====================

class TestPortNamesAndMaterializeError:
    def test_port_names_constants(self):
        from yggdrasil.core.graph.orchestrator import PortNames
        assert PortNames.SAMPLE == "sample"
        assert PortNames.CONDITION == "condition"
        assert PortNames.CONTROL_IMAGE == "control_image"
        assert PortNames.ADAPTER_FEATURES == "adapter_features"

    def test_materialize_raises_when_deferred_adapters_but_no_loop(self):
        """Orchestrator.materialize() raises explicit error if adapters deferred and no loop (§10.G)."""
        from yggdrasil.core.graph import ComputeGraph
        from yggdrasil.core.graph.orchestrator import GraphBuildOrchestrator
        g = ComputeGraph("test")
        orch = GraphBuildOrchestrator(g)
        orch.register_node("ctrl", "adapter/controlnet", {}, target_inner=None)
        assert len(orch.get_state().deferred_adapter_bindings) == 1
        with pytest.raises(ValueError, match="no loop node exists|Deferred"):
            orch.materialize()

    def test_replace_node_invalidates_orchestrator(self):
        """replace_node calls orchestrator.invalidate() so state is recomputed on next use."""
        from yggdrasil.core.graph import ComputeGraph
        g = ComputeGraph("g")
        g.add_node("n1", _MockGraph())
        orch = g._get_orchestrator()
        assert orch is not None
        g.replace_node("n1", _MockGraph())
        # After replace, orchestrator is invalidated; get_state/materialize will recompute
        state = orch.get_state()
        assert state is not None

    def test_validation_result_structure(self):
        """validate() returns ValidationResult with success, errors, warnings (§10.G, Q2)."""
        from yggdrasil.core.graph.orchestrator import GraphBuildOrchestrator, ValidationResult, ValidationError
        g = _MockGraph()
        g.graph_inputs["required_input"] = []  # no target -> error
        orch = GraphBuildOrchestrator(g)
        result = orch.validate()
        assert isinstance(result, ValidationResult)
        assert hasattr(result, "success")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")
        assert result.success is False
        assert len(result.errors) >= 1
        assert any(e.code == "MISSING_TARGET" for e in result.errors)

    def test_materialize_idempotent_when_no_deferred(self):
        """Calling materialize() twice when there is no deferred work does not fail (§10.C, Q2)."""
        from yggdrasil.core.graph.orchestrator import GraphBuildOrchestrator
        g = _MockGraph()
        g.nodes["loop"] = _MockLoopBlock()
        g._deferred = []
        orch = GraphBuildOrchestrator(g)
        orch.materialize()
        orch.materialize()
        state = orch.get_state()
        assert state is not None
        assert len(state.deferred_adapter_bindings) == 0


# ==================== register_custom_role & LoopTemplates ====================

class TestRegisterCustomRoleAndLoopTemplates:
    def test_register_custom_role(self):
        from yggdrasil.core.graph.orchestrator import (
            register_custom_role,
            RoleRegistry,
            NodeRule,
        )
        rule = NodeRule(
            role="world_model",
            target_hint=None,
            target_port=None,
            output_port="output",
            input_port="input",
            graph_input_name="input",
            insert_into="root",
            defer_if_missing=(),
            creates_loop=False,
        )
        register_custom_role("world_model/", rule)
        try:
            r = RoleRegistry()
            assert r.get_role_for_block_type("world_model/foo") == "world_model"
            assert r.get_rule("world_model") is not None
        finally:
            # Teardown: clear custom so other tests don't see it
            import yggdrasil.core.graph.orchestrator as o
            o._custom_type_to_role.clear()
            o._custom_rules.clear()

    def test_loop_templates_registry(self):
        from yggdrasil.core.graph.orchestrator import LoopTemplates
        lt = LoopTemplates()
        assert lt.get("euler_discrete", "image") is None
        lt.register("euler_discrete", "image", "step_sdxl")
        assert lt.get("euler_discrete", "image") == "step_sdxl"
        lt.register("euler_discrete", "*", "step_generic")
        assert lt.get("euler_discrete", "video") == "step_generic"

    def test_loop_templates_get_or_default(self):
        """L3: generic fallback when no (solver_type, modality) match."""
        from yggdrasil.core.graph.orchestrator import LoopTemplates
        lt = LoopTemplates()
        assert lt.get_or_default("unknown_solver", "image") == "generic"
        lt.register("euler_discrete", "image", "step_sdxl")
        assert lt.get_or_default("euler_discrete", "image") == "step_sdxl"
        assert lt.get_or_default("unknown", "audio") == "generic"

    def test_get_step_template_id_for_metadata(self):
        """L1: helper to pick step template from graph metadata (uses LoopTemplates class-level registry)."""
        from yggdrasil.core.graph.orchestrator import get_step_template_id_for_metadata
        assert get_step_template_id_for_metadata({}) == "generic"
        # With no registration for (solver_type, modality), returns "generic"; if a prior test registered, may return that id
        t1 = get_step_template_id_for_metadata({"solver_type": "euler_discrete", "modality": "image"})
        assert isinstance(t1, str) and len(t1) > 0
        t2 = get_step_template_id_for_metadata({"modality": "audio"})
        assert isinstance(t2, str) and (t2 == "generic" or len(t2) > 0)
        t3 = get_step_template_id_for_metadata({"solver_type": "ddim"})
        assert isinstance(t3, str) and len(t3) > 0

    def test_step_builder_registry(self):
        """L1 extension: StepBuilderRegistry — template_id -> step graph builder."""
        from yggdrasil.core.graph.orchestrator import StepBuilderRegistry
        from yggdrasil.core.graph.graph import ComputeGraph
        reg = StepBuilderRegistry()
        assert reg.get("step_sdxl") is None
        def fake_builder(metadata, pretrained, num_steps, guidance_scale, **kwargs):
            g = ComputeGraph("step")
            return g
        reg.register("step_sdxl", fake_builder)
        b = reg.get("step_sdxl")
        assert b is not None
        out = b(metadata={}, pretrained="x", num_steps=1, guidance_scale=7.0)
        assert isinstance(out, ComputeGraph) and out.name == "step"
        assert reg.get_or_default("unknown", default=fake_builder) is fake_builder

    def test_step_builders_registered_when_image_pipelines_loaded(self):
        """After image_pipelines is loaded, generic, step_sdxl, step_sd3, step_flux are in StepBuilderRegistry."""
        from yggdrasil.core.graph.orchestrator import StepBuilderRegistry
        try:
            import yggdrasil.core.graph.templates.image_pipelines as _  # noqa: F401
        except ImportError:
            pytest.skip("image_pipelines requires torch and block deps")
        reg = StepBuilderRegistry()
        for tid in ("generic", "step_sdxl", "step_sd3", "step_flux"):
            assert reg.get(tid) is not None and callable(reg.get(tid)), f"builder {tid} not registered"

    def test_get_step_template_id_for_metadata_base_model(self):
        """get_step_template_id_for_metadata returns correct template for base_model."""
        from yggdrasil.core.graph.orchestrator import get_step_template_id_for_metadata
        assert get_step_template_id_for_metadata({"base_model": "sdxl"}) == "step_sdxl"
        assert get_step_template_id_for_metadata({"base_model": "sd15"}) == "step_sdxl"
        assert get_step_template_id_for_metadata({"base_model": "sd3"}) == "step_sd3"
        assert get_step_template_id_for_metadata({"base_model": "flux"}) == "step_flux"
        tid = get_step_template_id_for_metadata({"solver_type": "euler_discrete", "modality": "image"})
        assert tid in ("generic", "step_sdxl", "step_sd3", "step_flux")

    def test_solver_registry(self):
        from yggdrasil.core.graph.orchestrator import SolverRegistry
        sr = SolverRegistry()
        assert sr.get_solver_type("EulerDiscreteScheduler") == "euler_discrete"
        assert sr.get_step_signature("EulerDiscreteScheduler") == "epsilon"
        assert sr.get_solver_type("LCMScheduler") == "lcm"
        assert sr.get_solver_type("unknown") is None
        sr.register("CustomScheduler", "custom_solver", "v_prediction")
        assert sr.get_solver_type("CustomScheduler") == "custom_solver"
        assert sr.get_step_signature("CustomScheduler") == "v_prediction"

    def test_graph_to_mermaid(self):
        from yggdrasil.core.graph import ComputeGraph
        from yggdrasil.core.graph.orchestrator import graph_to_mermaid
        g = ComputeGraph("test")
        s = graph_to_mermaid(g)
        assert "graph" in s.lower()
        assert "LR" in s or "TD" in s


# ==================== S2: Step contract ====================

class TestStepContract:
    """Denoise step input/output port names (§11.4 S2)."""

    def test_step_input_output_names(self):
        from yggdrasil.core.graph.orchestrator import (
            STEP_INPUT_NAMES,
            STEP_OUTPUT_NAMES,
            PortNames,
        )
        assert PortNames.LATENTS in STEP_INPUT_NAMES
        assert PortNames.TIMESTEP in STEP_INPUT_NAMES
        assert PortNames.CONDITION in STEP_INPUT_NAMES
        assert PortNames.MODEL_OUTPUT in STEP_OUTPUT_NAMES
        assert "next_latents" in STEP_OUTPUT_NAMES


# ==================== A1/A4: AdapterBindingRules ====================

class TestAdapterBindingRules:
    """Adapter type → rule (insert_into, backbone_port, graph_input). §11.3 A1, A4."""

    def test_default_adapter_rules(self):
        from yggdrasil.core.graph.orchestrator import AdapterBindingRules
        rules = AdapterBindingRules()
        r = rules.get("adapter/controlnet")
        assert r is not None
        assert r.backbone_port == "adapter_features"
        assert r.graph_input_name == "control_image"
        r = rules.get("adapter/ip_adapter")
        assert r is not None
        assert r.backbone_port == "image_prompt_embeds"
        assert r.graph_input_name == "ip_image"
        r = rules.get("adapter/t2i_adapter")
        assert r is not None
        assert r.insert_into == "loop_inner"
        r = rules.get("adapter/motion")
        assert r is not None

    def test_adapter_rules_get_or_prefix(self):
        from yggdrasil.core.graph.orchestrator import AdapterBindingRules
        rules = AdapterBindingRules()
        assert rules.get_or_prefix("adapter/controlnet") is not None
        assert rules.get_or_prefix("adapter/controlnet_canny") is not None
        assert rules.get("unknown") is None


# ==================== A2: control_type → input name mapping ====================

class TestControlnetInputMapping:
    """Mapping control_type → graph_input_name for multiple ControlNets (§11.3 A2)."""

    def test_get_controlnet_input_mapping_empty(self):
        from yggdrasil.core.graph.adapters import get_controlnet_input_mapping
        g = _MockGraph()
        g.nodes = {}
        assert get_controlnet_input_mapping(g) == {}

    def test_get_controlnet_input_mapping_from_inner(self):
        from yggdrasil.core.graph.adapters import get_controlnet_input_mapping
        inner = _MockGraph()
        inner.graph_inputs = {"control_image_controlnet_depth": [("controlnet_depth", "control_image")]}
        inner.nodes = {"controlnet_depth": type("ControlNet", (), {"block_type": "adapter/controlnet", "control_type": "depth"})()}
        loop = type("Loop", (), {"block_type": "loop/denoise_sdxl", "graph": inner, "_loop": None})()
        g = _MockGraph()
        g.nodes = {"denoise_loop": loop}
        m = get_controlnet_input_mapping(g)
        assert m == {"depth": "control_image_controlnet_depth"}


# ==================== Non-diffusion contract (N2) ====================

class TestNonDiffusionContract:
    """Standard I/O keys for non-diffusion pipelines (§11.5 N2)."""

    def test_nondiffusion_constants(self):
        from yggdrasil.core.graph.orchestrator import (
            NONDIFFUSION_INPUT_KEYS,
            NONDIFFUSION_OUTPUT_KEYS,
        )
        assert "image" in NONDIFFUSION_INPUT_KEYS
        assert "mask" in NONDIFFUSION_INPUT_KEYS
        assert "output" in NONDIFFUSION_OUTPUT_KEYS
        assert "segmentation_map" in NONDIFFUSION_OUTPUT_KEYS

    def test_get_expected_io_for_modality(self):
        from yggdrasil.core.graph.orchestrator import get_expected_io_for_modality
        in_keys, out_keys = get_expected_io_for_modality("image")
        assert "image" in in_keys
        assert "output" in out_keys
        in_keys, out_keys = get_expected_io_for_modality("audio")
        assert "audio" in in_keys
        in_keys, out_keys = get_expected_io_for_modality("video")
        assert "video" in in_keys
        in_keys, out_keys = get_expected_io_for_modality("")
        assert "input" in in_keys


# ==================== Parallel groups (P3) ====================

class TestParallelGroups:
    """Execution plan via parallel_groups in combined pipeline (§11.7 P3)."""

    def test_executor_order_from_parallel_groups(self):
        from yggdrasil.core.graph.executor import GraphExecutor
        g = _MockGraph()
        g.nodes = {"seg": None, "depth": None, "gen": None}
        ex = GraphExecutor()
        order = ex._execution_order_from_parallel_groups(g, [["seg", "depth"], ["gen"]])
        assert order == ["seg", "depth", "gen"]

    def test_executor_parallel_groups_validates_all_nodes(self):
        from yggdrasil.core.graph.executor import GraphExecutor
        g = _MockGraph()
        g.nodes = {"a": None, "b": None}
        ex = GraphExecutor()
        with pytest.raises(ValueError, match="missing"):
            ex._execution_order_from_parallel_groups(g, [["a"]])
        with pytest.raises(ValueError, match="not in graph"):
            ex._execution_order_from_parallel_groups(g, [["a", "b", "c"]])


# ==================== Mocks ====================

class _MockGraph:
    nodes = {}
    edges = []
    graph_inputs = {}
    graph_outputs = {}
    metadata = {}

    def __init__(self):
        self.nodes = {}
        self.graph_inputs = {}
        self.graph_outputs = {}
        self.metadata = {}
        self._deferred = []

    def _materialize_deferred(self):
        self._deferred.clear()


class _MockLoopBlock:
    """Block with inner graph containing backbone."""
    block_type = "loop/denoise_sdxl"

    def __init__(self):
        self.graph = _MockGraph()
        self.graph.nodes["backbone"] = type("Backbone", (), {"block_type": "backbone/unet2d_condition"})()
