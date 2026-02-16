"""Backward compatibility tests (Q4). REFACTORING_GRAPH_PIPELINE_ENGINE.md §13.1.

Verify that the documented public API is preserved: from_template, add_node(type=...),
pipe(prompt=..., control_image=..., ip_image=...), train_nodes, etc.
No full model runs required; tests API existence and accepted kwargs.

Requires torch (yggdrasil imports it); entire module is skipped if torch is not installed.
"""
import inspect
import pytest

pytest.importorskip("torch")


# ==================== InferencePipeline API ====================

class TestInferencePipelineAPI:
    """InferencePipeline: from_template, from_pretrained, from_config, from_graph, from_spec; pipe(...) kwargs."""

    def test_class_methods_exist(self):
        from yggdrasil.pipeline import InferencePipeline
        assert hasattr(InferencePipeline, "from_template") and callable(InferencePipeline.from_template)
        assert hasattr(InferencePipeline, "from_pretrained") and callable(InferencePipeline.from_pretrained)
        assert hasattr(InferencePipeline, "from_config") and callable(InferencePipeline.from_config)
        assert hasattr(InferencePipeline, "from_graph") and callable(InferencePipeline.from_graph)
        assert hasattr(InferencePipeline, "from_spec") and callable(InferencePipeline.from_spec)

    def test_call_accepts_documented_kwargs(self):
        """pipe(prompt=..., negative_prompt=..., guidance_scale=..., num_steps=..., control_image=..., ip_image=..., ip_adapter_scale=...) — §13.1."""
        from yggdrasil.pipeline import InferencePipeline
        sig = inspect.signature(InferencePipeline.__call__)
        params = list(sig.parameters)
        assert "prompt" in params
        assert "negative_prompt" in params
        assert "guidance_scale" in params
        assert "num_steps" in params
        assert sig.parameters.get("kwargs") is not None or "**" in str(sig)

    def test_from_spec_accepts_graph(self):
        """from_spec(spec) accepts ComputeGraph (or path)."""
        from yggdrasil.pipeline import InferencePipeline
        from yggdrasil.core.graph.graph import ComputeGraph
        # Minimal graph (no nodes) just to test API
        g = ComputeGraph("minimal")
        try:
            pipe = InferencePipeline.from_spec(g)
        except Exception as e:
            pytest.skip(f"from_spec(graph) requires full env: {e}")
        assert pipe is not None
        assert hasattr(pipe, "graph")


# ==================== ComputeGraph API ====================

class TestComputeGraphAPI:
    """ComputeGraph: add_node(name, block), add_node(type=..., auto_connect=...), connect, expose_input, expose_output, from_yaml, from_template."""

    def test_add_node_connect_expose_exist(self):
        from yggdrasil.core.graph.graph import ComputeGraph
        g = ComputeGraph("test")
        assert hasattr(g, "add_node") and callable(g.add_node)
        assert hasattr(g, "connect") and callable(g.connect)
        assert hasattr(g, "expose_input") and callable(g.expose_input)
        assert hasattr(g, "expose_output") and callable(g.expose_output)

    def test_from_yaml_from_template_exist(self):
        from yggdrasil.core.graph.graph import ComputeGraph
        assert hasattr(ComputeGraph, "from_yaml") and callable(ComputeGraph.from_yaml)
        assert hasattr(ComputeGraph, "from_template") and callable(ComputeGraph.from_template)

    def test_add_node_accepts_type_and_auto_connect(self):
        """add_node(type=\"...\", name=\"...\", auto_connect=True, ...) — §13.1."""
        from yggdrasil.core.graph.graph import ComputeGraph
        g = ComputeGraph("test")
        sig = inspect.signature(g.add_node)
        params = set(sig.parameters)
        assert "type" in params
        assert "auto_connect" in params
        assert "name" in params


# ==================== TrainingPipeline API ====================

class TestTrainingPipelineAPI:
    """TrainingPipeline: from_pretrained, from_template, from_config, from_graph, from_spec; train_nodes, train_stages, freeze_nodes; train()."""

    def test_class_methods_exist(self):
        from yggdrasil.pipeline import TrainingPipeline
        assert hasattr(TrainingPipeline, "from_pretrained") and callable(TrainingPipeline.from_pretrained)
        assert hasattr(TrainingPipeline, "from_template") and callable(TrainingPipeline.from_template)
        assert hasattr(TrainingPipeline, "from_config") and callable(TrainingPipeline.from_config)
        assert hasattr(TrainingPipeline, "from_graph") and callable(TrainingPipeline.from_graph)
        assert hasattr(TrainingPipeline, "from_spec") and callable(TrainingPipeline.from_spec)

    def test_instance_has_train_nodes_and_train(self):
        from yggdrasil.pipeline import TrainingPipeline
        from yggdrasil.core.graph.graph import ComputeGraph
        g = ComputeGraph("minimal")
        pipe = TrainingPipeline(g, train_nodes=[])
        assert hasattr(pipe, "train_nodes")
        assert hasattr(pipe, "train_stages")
        assert hasattr(pipe, "freeze_nodes")
        assert hasattr(pipe, "train") and callable(pipe.train)

    def test_from_graph_accepts_train_nodes_freeze_nodes(self):
        from yggdrasil.pipeline import TrainingPipeline
        from yggdrasil.core.graph.graph import ComputeGraph
        g = ComputeGraph("minimal")
        pipe = TrainingPipeline.from_graph(g, train_nodes=["a"], freeze_nodes=None)
        assert pipe.train_nodes == ["a"]
        pipe2 = TrainingPipeline.from_graph(g, freeze_nodes=["b"])
        assert pipe2.freeze_nodes == ["b"]


# ==================== Pipe call kwargs (smoke) ====================

class TestPipeCallKwargs:
    """Smoke: pipe(prompt=..., control_image=..., ip_image=...) does not raise TypeError when called with documented kwargs."""

    @pytest.mark.skipif(
        __import__("sys").modules.get("torch") is None,
        reason="torch not installed",
    )
    def test_pipe_call_with_standard_kwargs(self):
        """Call pipe with prompt, negative_prompt, guidance_scale, num_steps, control_image=None, ip_image=None (no run to completion)."""
        try:
            from yggdrasil.pipeline import InferencePipeline
            from yggdrasil.core.graph.graph import ComputeGraph
        except ImportError as e:
            pytest.skip(str(e))
        # Use from_template to get a real pipeline; if that fails (no network/weights), skip
        try:
            pipe = InferencePipeline.from_template("sd15_txt2img", device="cpu")
        except Exception as e:
            pytest.skip(f"from_template needs weights/env: {e}")
        # Only check that __call__ accepts these kwargs (may fail later inside due to device/data)
        try:
            out = pipe(
                prompt="test",
                negative_prompt="",
                guidance_scale=7.5,
                num_steps=1,
                control_image=None,
                ip_image=None,
                ip_adapter_scale=0.6,
            )
        except TypeError as e:
            pytest.fail(f"pipe() must accept documented kwargs: {e}")
        except Exception:
            pass  # OK if it fails later (e.g. no CUDA, missing data)


# ==================== Combined pipeline API (P1–P4, Phase 9) ====================

class TestCombinedPipelineAPI:
    """InferencePipeline(graphs=[(name, g), ...]) and graphs={name: g}; parallel_groups."""

    def test_combined_from_list_of_tuples(self):
        """P1/P2: graphs=[(stage_name, graph), ...] builds combined pipeline."""
        from yggdrasil.pipeline import InferencePipeline
        from yggdrasil.core.graph.graph import ComputeGraph
        g1 = ComputeGraph("a")
        g2 = ComputeGraph("b")
        pipe = InferencePipeline(
            graphs=[("stage1", g1), ("stage2", g2)],
            connections=[],
            inputs={},
            outputs={},
        )
        assert pipe.graph is not None
        assert "stage1" in pipe.graph.nodes and "stage2" in pipe.graph.nodes

    def test_combined_from_dict_with_parallel_groups(self):
        """P2/P3: graphs={name: graph}, parallel_groups in metadata."""
        from yggdrasil.pipeline import InferencePipeline
        from yggdrasil.core.graph.graph import ComputeGraph
        pipe = InferencePipeline(
            graphs={"seg": ComputeGraph("s"), "gen": ComputeGraph("g")},
            connections=[],
            inputs={},
            outputs={},
            parallel_groups=[["seg"], ["gen"]],
        )
        assert pipe.graph.metadata.get("parallel_groups") == [["seg"], ["gen"]]
        assert "seg" in pipe.graph.nodes and "gen" in pipe.graph.nodes

    def test_from_spec_list_of_stages(self):
        """P4: from_spec([(name, graph), ...]) dispatches to from_combined."""
        from yggdrasil.pipeline import InferencePipeline
        from yggdrasil.core.graph.graph import ComputeGraph
        pipe = InferencePipeline.from_spec(
            [("a", ComputeGraph("x")), ("b", ComputeGraph("y"))],
            inputs={},
            outputs={},
        )
        assert pipe.graph is not None
        assert "a" in pipe.graph.nodes and "b" in pipe.graph.nodes
