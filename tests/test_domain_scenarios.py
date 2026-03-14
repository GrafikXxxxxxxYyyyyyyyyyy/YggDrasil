"""Domain scenario integration tests."""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

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
    AbstractInnerModule,
)
from yggdrasill.workflow.workflow import Workflow


class FakeVAEEncoder(AbstractConverter):
    @property
    def block_type(self) -> str:
        return "diffusion/vae_encoder"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": f"latent({inputs.get('input')})"}


class FakeVAEDecoder(AbstractConverter):
    @property
    def block_type(self) -> str:
        return "diffusion/vae_decoder"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": f"decoded({inputs.get('input', '')})"}


class FakeCLIPEncoder(AbstractConjector):
    @property
    def block_type(self) -> str:
        return "diffusion/clip_encoder"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"condition": f"clip({inputs.get('input')})"}


class FakeUNet(AbstractBackbone):
    @property
    def block_type(self) -> str:
        return "diffusion/unet"

    def declare_ports(self) -> List[Port]:
        return [
            Port("latent", PortDirection.IN, PortType.ANY),
            Port("conditioning", PortDirection.IN, PortType.ANY),
            Port("pred", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"pred": f"unet({inputs.get('latent')},{inputs.get('conditioning')})"}


class FakeScheduler(AbstractHelper):
    @property
    def block_type(self) -> str:
        return "diffusion/scheduler"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": f"scheduled({inputs.get('query', '')})"}


class FakeLLM(AbstractBackbone):
    @property
    def block_type(self) -> str:
        return "llm/backbone"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.ANY),
            Port("output", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": f"llm({inputs.get('input')})"}


class FakeTokenizer(AbstractConverter):
    @property
    def block_type(self) -> str:
        return "llm/tokenizer"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": f"tokens({inputs.get('input')})"}


class FakeDetokenizer(AbstractConverter):
    @property
    def block_type(self) -> str:
        return "llm/detokenizer"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": f"text({inputs.get('input')})"}


class FakeRouter(AbstractInnerModule):
    @property
    def block_type(self) -> str:
        return "agent/router"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.ANY),
            Port("output", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": f"route({inputs.get('input')})"}


class FakeAgentWithToolCalls(AbstractInnerModule):
    """Agent node that returns tool_calls in its output, testing the contract."""

    @property
    def block_type(self) -> str:
        return "agent/with_tool_calls"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.ANY),
            Port("output", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "output": {
                "response": f"thinking about {inputs.get('input')}",
                "tool_calls": [
                    {"tool_id": "search", "args": {"query": "test"}},
                    {"tool_id": "calc", "args": {"expr": "2+2"}},
                ],
            }
        }


class FakeToolExecutor(AbstractConjector):
    @property
    def block_type(self) -> str:
        return "agent/tool_exec"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"condition": f"tool_result({inputs.get('input')})"}


class FakeMemory(AbstractHelper):
    @property
    def block_type(self) -> str:
        return "agent/memory"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": f"mem({inputs.get('query', '')})"}


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
    r.register("agent/with_tool_calls", FakeAgentWithToolCalls)
    return r


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
                {"source_node": "vae_enc", "source_port": "output", "target_node": "unet", "target_port": "latent"},
                {"source_node": "clip", "source_port": "condition", "target_node": "unet", "target_port": "conditioning"},
                {"source_node": "unet", "source_port": "pred", "target_node": "sched", "target_port": "query"},
                {"source_node": "sched", "source_port": "result", "target_node": "vae_dec", "target_port": "input"},
            ],
            "exposed_inputs": [
                {"node_id": "vae_enc", "port_name": "input", "name": "image"},
                {"node_id": "clip", "port_name": "input", "name": "prompt"},
            ],
            "exposed_outputs": [
                {"node_id": "vae_dec", "port_name": "output", "name": "output"},
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
                {"source_node": "enc", "source_port": "output", "target_node": "dec", "target_port": "input"},
            ],
            "exposed_inputs": [{"node_id": "enc", "port_name": "input", "name": "x"}],
            "exposed_outputs": [{"node_id": "dec", "port_name": "output", "name": "y"}],
        }
        h1 = Hypergraph.from_config(cfg, registry=domain_registry)
        out1 = h1.run({"x": "pixel"})
        save_hypergraph(h1, tmp_path / "diff")
        h2 = load_hypergraph(tmp_path / "diff", registry=domain_registry)
        out2 = h2.run({"x": "pixel"})
        assert out1 == out2


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
                {"source_node": "tok", "source_port": "output", "target_node": "llm", "target_port": "input"},
                {"source_node": "llm", "source_port": "output", "target_node": "detok", "target_port": "input"},
            ],
            "exposed_inputs": [{"node_id": "tok", "port_name": "input", "name": "text"}],
            "exposed_outputs": [{"node_id": "detok", "port_name": "output", "name": "response"}],
        }
        h = Hypergraph.from_config(cfg, registry=domain_registry)
        out = h.run({"text": "hello world"})
        assert "response" in out
        assert "llm" in out["response"]
        assert "tokens" in out["response"]

    def test_llm_workflow_two_steps(self, domain_registry):
        clear_plan_cache()
        llm_cfg = {
            "graph_id": "llm_step",
            "nodes": [
                {"node_id": "tok", "block_type": "llm/tokenizer"},
                {"node_id": "llm", "block_type": "llm/backbone"},
                {"node_id": "detok", "block_type": "llm/detokenizer"},
            ],
            "edges": [
                {"source_node": "tok", "source_port": "output", "target_node": "llm", "target_port": "input"},
                {"source_node": "llm", "source_port": "output", "target_node": "detok", "target_port": "input"},
            ],
            "exposed_inputs": [{"node_id": "tok", "port_name": "input", "name": "input"}],
            "exposed_outputs": [{"node_id": "detok", "port_name": "output", "name": "output"}],
        }
        step1 = Hypergraph.from_config(llm_cfg, registry=domain_registry)
        step2 = Hypergraph.from_config(llm_cfg, registry=domain_registry)

        w = Workflow(workflow_id="llm_chain")
        w.add_node("summarize", step1)
        w.add_node("refine", step2)
        w.add_edge("summarize", "output", "refine", "input")
        w.expose_input("summarize", "input", "input")
        w.expose_output("refine", "output", "output")

        out = w.run({"input": "long document"}, validate_before=False)
        assert "output" in out
        assert "llm" in out["output"]


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
                {"source_node": "router", "source_port": "output", "target_node": "tool", "target_port": "input"},
                {"source_node": "tool", "source_port": "condition", "target_node": "mem", "target_port": "query"},
                {"source_node": "mem", "source_port": "result", "target_node": "router", "target_port": "input"},
            ],
            "exposed_inputs": [{"node_id": "router", "port_name": "input", "name": "query"}],
            "exposed_outputs": [{"node_id": "mem", "port_name": "result", "name": "answer"}],
        }
        h = Hypergraph.from_config(cfg, registry=domain_registry)
        out = h.run({"query": "what is 2+2?"}, num_loop_steps=3, validate_before=False)
        assert "answer" in out
        assert "route" in out["answer"]
        assert "tool_result" in out["answer"]
        assert "mem" in out["answer"]

    def test_agent_tool_calls_contract(self, domain_registry):
        """Verify that an agent node can return tool_calls in its output dict."""
        clear_plan_cache()
        h = Hypergraph()
        h.add_node_from_config("agent", "agent/with_tool_calls", registry=domain_registry)
        h.expose_input("agent", "input", "query")
        h.expose_output("agent", "output", "result")
        out = h.run({"query": "what is 2+2?"}, validate_before=False)
        result = out["result"]
        assert isinstance(result, dict)
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["tool_id"] == "search"
        assert result["tool_calls"][1]["tool_id"] == "calc"

    def test_agent_workflow_with_cycle(self, domain_registry):
        clear_plan_cache()
        agent_cfg = {
            "graph_id": "agent_step",
            "metadata": {"num_loop_steps": 2},
            "nodes": [
                {"node_id": "router", "block_type": "agent/router"},
                {"node_id": "tool", "block_type": "agent/tool_exec"},
            ],
            "edges": [
                {"source_node": "router", "source_port": "output", "target_node": "tool", "target_port": "input"},
                {"source_node": "tool", "source_port": "condition", "target_node": "router", "target_port": "input"},
            ],
            "exposed_inputs": [{"node_id": "router", "port_name": "input", "name": "input"}],
            "exposed_outputs": [{"node_id": "tool", "port_name": "condition", "name": "condition"}],
        }
        step1 = Hypergraph.from_config(agent_cfg, registry=domain_registry)
        step2 = Hypergraph.from_config(agent_cfg, registry=domain_registry)

        w = Workflow(workflow_id="agent_wf")
        w.add_node("think", step1)
        w.add_node("act", step2)
        w.add_edge("think", "condition", "act", "input")
        w.expose_input("think", "input", "q")
        w.expose_output("act", "condition", "a")

        result = w.run({"q": "plan"}, validate_before=False)
        assert "a" in result
        assert "route" in result["a"]
