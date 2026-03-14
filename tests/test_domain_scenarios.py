"""Domain scenario integration tests.

These verify that the YggDrasill architecture can express three canonical
patterns -- diffusion pipeline, LLM chain, and agent loop -- using stubs
(no real models, only identity-like transformations).
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from yggdrasill.engine.edge import Edge
from yggdrasill.engine.planner import clear_plan_cache
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.foundation.registry import BlockRegistry
from yggdrasill.hypergraph.serialization import load_hypergraph, save_hypergraph
from yggdrasill.task_nodes.abstract import (
    AbstractBackbone,
    AbstractConjector,
    AbstractConverter,
    AbstractHelper,
    AbstractInjector,
    AbstractInnerModule,
)
from yggdrasill.workflow.workflow import Workflow


# ====================================================================
# Stub blocks for domain scenarios (registered into a local registry)
# ====================================================================

class FakeVAEEncoder(AbstractConverter):
    @property
    def block_type(self) -> str:
        return "diffusion/vae_encoder"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": f"latent({inputs.get('in')})"}


class FakeVAEDecoder(AbstractConverter):
    @property
    def block_type(self) -> str:
        return "diffusion/vae_decoder"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raw = inputs.get("in", "")
        return {"out": f"decoded({raw})"}


class FakeCLIPEncoder(AbstractInjector):
    @property
    def block_type(self) -> str:
        return "diffusion/clip_encoder"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": f"clip({inputs.get('condition')})"}


class FakeUNet(AbstractBackbone):
    @property
    def block_type(self) -> str:
        return "diffusion/unet"

    def declare_ports(self) -> List[Port]:
        return [
            Port("latent", PortDirection.IN, PortType.ANY),
            Port("conditioning", PortDirection.IN, PortType.ANY),
            Port("out", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": f"unet({inputs.get('latent')},{inputs.get('conditioning')})"}


class FakeScheduler(AbstractHelper):
    @property
    def block_type(self) -> str:
        return "diffusion/scheduler"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": f"scheduled({inputs.get('in', '')})"}


class FakeLLM(AbstractBackbone):
    @property
    def block_type(self) -> str:
        return "llm/backbone"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": f"llm({inputs.get('in')})"}


class FakeTokenizer(AbstractConverter):
    @property
    def block_type(self) -> str:
        return "llm/tokenizer"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": f"tokens({inputs.get('in')})"}


class FakeDetokenizer(AbstractConverter):
    @property
    def block_type(self) -> str:
        return "llm/detokenizer"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": f"text({inputs.get('in')})"}


class FakeRouter(AbstractInnerModule):
    """Agent router: decides next action based on input."""

    @property
    def block_type(self) -> str:
        return "agent/router"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": f"route({inputs.get('in')})"}


class FakeToolExecutor(AbstractConjector):
    """Agent tool execution stub."""

    @property
    def block_type(self) -> str:
        return "agent/tool_exec"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": f"tool_result({inputs.get('in')})"}


class FakeMemory(AbstractHelper):
    @property
    def block_type(self) -> str:
        return "agent/memory"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": f"mem({inputs.get('in', '')})"}


@pytest.fixture()
def domain_registry():
    r = BlockRegistry()
    r.register("diffusion/vae_encoder", FakeVAEEncoder)
    r.register("diffusion/vae_decoder", FakeVAEDecoder)
    r.register("diffusion/clip_encoder", FakeCLIPEncoder)
    r.register("diffusion/unet", FakeUNet)
    r.register("diffusion/scheduler", FakeScheduler)
    r.register("llm/backbone", FakeLLM)
    r.register("llm/tokenizer", FakeTokenizer)
    r.register("llm/detokenizer", FakeDetokenizer)
    r.register("agent/router", FakeRouter)
    r.register("agent/tool_exec", FakeToolExecutor)
    r.register("agent/memory", FakeMemory)
    return r


# ====================================================================
# Scenario 1: Diffusion Pipeline
#   CLIP -> UNet (with latent from VAE encoder) -> Scheduler -> VAE Decoder
# ====================================================================

class TestDiffusionPipeline:
    def test_diffusion_hypergraph(self, domain_registry):
        clear_plan_cache()
        cfg = {
            "graph_id": "diffusion_pipeline",
            "graph_kind": "task_hypergraph",
            "nodes": [
                {"node_id": "vae_enc", "block_type": "diffusion/vae_encoder"},
                {"node_id": "clip", "block_type": "diffusion/clip_encoder"},
                {"node_id": "unet", "block_type": "diffusion/unet"},
                {"node_id": "sched", "block_type": "diffusion/scheduler"},
                {"node_id": "vae_dec", "block_type": "diffusion/vae_decoder"},
            ],
            "edges": [
                {"source_node": "vae_enc", "source_port": "out", "target_node": "unet", "target_port": "latent"},
                {"source_node": "clip", "source_port": "out", "target_node": "unet", "target_port": "conditioning"},
                {"source_node": "unet", "source_port": "out", "target_node": "sched", "target_port": "in"},
                {"source_node": "sched", "source_port": "out", "target_node": "vae_dec", "target_port": "in"},
            ],
            "exposed_inputs": [
                {"node_id": "vae_enc", "port_name": "in", "name": "image"},
                {"node_id": "clip", "port_name": "condition", "name": "prompt"},
            ],
            "exposed_outputs": [
                {"node_id": "vae_dec", "port_name": "out", "name": "output"},
            ],
        }
        h = Hypergraph.from_config(cfg, registry=domain_registry)
        out = h.run({"image": "img_data", "prompt": "a cat"})
        assert "output" in out
        assert "unet" in out["output"]
        assert "clip" in out["output"]
        assert "latent" in out["output"]

    def test_diffusion_serialization_roundtrip(self, tmp_path, domain_registry):
        clear_plan_cache()
        cfg = {
            "graph_id": "diff_ser",
            "nodes": [
                {"node_id": "enc", "block_type": "diffusion/vae_encoder"},
                {"node_id": "dec", "block_type": "diffusion/vae_decoder"},
            ],
            "edges": [
                {"source_node": "enc", "source_port": "out", "target_node": "dec", "target_port": "in"},
            ],
            "exposed_inputs": [{"node_id": "enc", "port_name": "in", "name": "x"}],
            "exposed_outputs": [{"node_id": "dec", "port_name": "out", "name": "y"}],
        }
        h1 = Hypergraph.from_config(cfg, registry=domain_registry)
        out1 = h1.run({"x": "pixel"})
        save_hypergraph(h1, tmp_path / "diff")
        h2 = load_hypergraph(tmp_path / "diff", registry=domain_registry)
        out2 = h2.run({"x": "pixel"})
        assert out1 == out2


# ====================================================================
# Scenario 2: LLM Chain
#   Tokenizer -> LLM -> Detokenizer   (two-step workflow: summarize then refine)
# ====================================================================

class TestLLMChain:
    def test_llm_hypergraph(self, domain_registry):
        clear_plan_cache()
        cfg = {
            "graph_id": "llm_pipe",
            "nodes": [
                {"node_id": "tok", "block_type": "llm/tokenizer"},
                {"node_id": "llm", "block_type": "llm/backbone"},
                {"node_id": "detok", "block_type": "llm/detokenizer"},
            ],
            "edges": [
                {"source_node": "tok", "source_port": "out", "target_node": "llm", "target_port": "in"},
                {"source_node": "llm", "source_port": "out", "target_node": "detok", "target_port": "in"},
            ],
            "exposed_inputs": [{"node_id": "tok", "port_name": "in", "name": "text"}],
            "exposed_outputs": [{"node_id": "detok", "port_name": "out", "name": "response"}],
        }
        h = Hypergraph.from_config(cfg, registry=domain_registry)
        out = h.run({"text": "hello world"})
        assert "response" in out
        assert "llm" in out["response"]
        assert "tokens" in out["response"]

    def test_llm_workflow_two_steps(self, domain_registry):
        """Workflow: summarize_step -> refine_step, both are LLM hypergraphs."""
        clear_plan_cache()
        llm_cfg = {
            "graph_id": "llm_step",
            "nodes": [
                {"node_id": "tok", "block_type": "llm/tokenizer"},
                {"node_id": "llm", "block_type": "llm/backbone"},
                {"node_id": "detok", "block_type": "llm/detokenizer"},
            ],
            "edges": [
                {"source_node": "tok", "source_port": "out", "target_node": "llm", "target_port": "in"},
                {"source_node": "llm", "source_port": "out", "target_node": "detok", "target_port": "in"},
            ],
            "exposed_inputs": [{"node_id": "tok", "port_name": "in", "name": "in"}],
            "exposed_outputs": [{"node_id": "detok", "port_name": "out", "name": "out"}],
        }
        step1 = Hypergraph.from_config(llm_cfg, registry=domain_registry)
        step2 = Hypergraph.from_config(llm_cfg, registry=domain_registry)

        w = Workflow(workflow_id="llm_chain")
        w.add_node("summarize", step1)
        w.add_node("refine", step2)
        w.add_edge(Edge("summarize", "out", "refine", "in"))
        w.expose_input("summarize", "in", "input")
        w.expose_output("refine", "out", "output")

        out = w.run({"input": "long document"}, validate_before=False)
        assert "output" in out
        assert "llm" in out["output"]


# ====================================================================
# Scenario 3: Agent Loop
#   Router -> ToolExec -> Memory -> Router (cycle with num_loop_steps)
# ====================================================================

class TestAgentLoop:
    def test_agent_hypergraph_with_cycle(self, domain_registry):
        clear_plan_cache()
        cfg = {
            "graph_id": "agent_loop",
            "metadata": {"num_loop_steps": 3},
            "nodes": [
                {"node_id": "router", "block_type": "agent/router"},
                {"node_id": "tool", "block_type": "agent/tool_exec"},
                {"node_id": "mem", "block_type": "agent/memory"},
            ],
            "edges": [
                {"source_node": "router", "source_port": "out", "target_node": "tool", "target_port": "in"},
                {"source_node": "tool", "source_port": "out", "target_node": "mem", "target_port": "in"},
                {"source_node": "mem", "source_port": "out", "target_node": "router", "target_port": "in"},
            ],
            "exposed_inputs": [{"node_id": "router", "port_name": "in", "name": "query"}],
            "exposed_outputs": [{"node_id": "mem", "port_name": "out", "name": "answer"}],
        }
        h = Hypergraph.from_config(cfg, registry=domain_registry)
        out = h.run({"query": "what is 2+2?"}, num_loop_steps=3, validate_before=False)
        assert "answer" in out
        assert "route" in out["answer"]
        assert "tool_result" in out["answer"]
        assert "mem" in out["answer"]

    def test_agent_workflow_with_cycle(self, domain_registry):
        """Two agent stages in a workflow, one with an internal cycle."""
        clear_plan_cache()
        agent_cfg = {
            "graph_id": "agent_step",
            "metadata": {"num_loop_steps": 2},
            "nodes": [
                {"node_id": "router", "block_type": "agent/router"},
                {"node_id": "tool", "block_type": "agent/tool_exec"},
            ],
            "edges": [
                {"source_node": "router", "source_port": "out", "target_node": "tool", "target_port": "in"},
                {"source_node": "tool", "source_port": "out", "target_node": "router", "target_port": "in"},
            ],
            "exposed_inputs": [{"node_id": "router", "port_name": "in", "name": "in"}],
            "exposed_outputs": [{"node_id": "tool", "port_name": "out", "name": "out"}],
        }
        step1 = Hypergraph.from_config(agent_cfg, registry=domain_registry)
        step2 = Hypergraph.from_config(agent_cfg, registry=domain_registry)

        w = Workflow(workflow_id="agent_wf")
        w.add_node("think", step1)
        w.add_node("act", step2)
        w.add_edge(Edge("think", "out", "act", "in"))
        w.expose_input("think", "in", "q")
        w.expose_output("act", "out", "a")

        result = w.run({"q": "plan"}, validate_before=False)
        assert "a" in result
        assert "route" in result["a"]
