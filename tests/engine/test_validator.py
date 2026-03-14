from typing import Any, Dict, List

from yggdrasill.engine.validator import validate
from tests.engine.helpers import make_chain, make_cycle
from yggdrasill.engine.edge import Edge
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.foundation.block import AbstractBaseBlock
from yggdrasill.foundation.node import AbstractGraphNode
from yggdrasill.foundation.port import Port, PortDirection, PortType
from tests.foundation.helpers import IdentityTaskNode


class _TypedNode(AbstractBaseBlock, AbstractGraphNode):
    """Node with a configurable single output type."""

    def __init__(self, node_id: str, in_type: PortType, out_type: PortType) -> None:
        AbstractBaseBlock.__init__(self)
        AbstractGraphNode.__init__(self, node_id=node_id)
        self._in_type = in_type
        self._out_type = out_type

    @property
    def block_type(self) -> str:
        return "test/typed"

    def declare_ports(self) -> List[Port]:
        return [
            Port("in", PortDirection.IN, self._in_type),
            Port("out", PortDirection.OUT, self._out_type),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": inputs.get("in")}


class TestValidatorValid:
    def test_chain_valid(self):
        h = make_chain("A", "B", "C")
        r = validate(h)
        assert r.valid

    def test_cycle_valid(self):
        h = make_cycle()
        r = validate(h)
        assert r.valid


class TestValidatorErrors:
    def test_missing_port(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        h._edges.append(Edge("A", "nonexistent", "B", "in"))
        r = validate(h)
        assert not r.valid
        assert any("nonexistent" in e for e in r.errors)

    def test_required_input_uncovered(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        r = validate(h)
        assert not r.valid
        assert any("Required input" in e for e in r.errors)


class TestValidatorIncompatiblePorts:
    def test_incompatible_types_error(self):
        h = Hypergraph()
        h.add_node("A", _TypedNode("A", PortType.ANY, PortType.TENSOR))
        h.add_node("B", _TypedNode("B", PortType.IMAGE, PortType.ANY))
        # Inject edge directly to bypass add_edge's own compatibility check
        edge = Edge("A", "out", "B", "in")
        h._edges.append(edge)
        h._in_edges.setdefault("B", []).append(edge)
        h._out_edges.setdefault("A", []).append(edge)
        h.expose_output("B", "out", "y")
        r = validate(h)
        assert not r.valid
        assert any("Incompatible" in e for e in r.errors)

    def test_compatible_types_ok(self):
        h = Hypergraph()
        h.add_node("A", _TypedNode("A", PortType.ANY, PortType.TENSOR))
        h.add_node("B", _TypedNode("B", PortType.TENSOR, PortType.ANY))
        h.add_edge(Edge("A", "out", "B", "in"))
        h.expose_input("A", "in", "x")
        h.expose_output("B", "out", "y")
        r = validate(h)
        assert r.valid


class TestValidatorExposedErrors:
    def test_exposed_input_unknown_node(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.expose_input("A", "in", "x")
        h.expose_output("A", "out", "y")
        h._exposed_inputs.append({"node_id": "GHOST", "port_name": "in"})
        r = validate(h)
        assert not r.valid
        assert any("GHOST" in e for e in r.errors)

    def test_exposed_output_unknown_port(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.expose_input("A", "in", "x")
        h._exposed_outputs.append({"node_id": "A", "port_name": "nonexistent"})
        r = validate(h)
        assert not r.valid
        assert any("nonexistent" in e for e in r.errors)


class TestValidatorCycleWarning:
    def test_cycle_without_num_loop_steps_warns(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        h.add_edge(Edge("A", "out", "B", "in"))
        h.add_edge(Edge("B", "out", "A", "in"))
        h.expose_input("A", "in", "x")
        h.expose_output("B", "out", "y")
        r = validate(h)
        assert r.valid
        assert len(r.warnings) >= 1
        assert any("num_loop_steps" in w for w in r.warnings)

    def test_cycle_with_num_loop_steps_no_warning(self):
        h = make_cycle()
        r = validate(h)
        assert r.valid
        assert len(r.warnings) == 0

    def test_dag_no_warning(self):
        h = make_chain("A", "B", "C")
        r = validate(h)
        assert r.valid
        assert len(r.warnings) == 0

    def test_self_loop_without_num_loop_steps_warns(self):
        h = Hypergraph()
        h.add_node("X", IdentityTaskNode(node_id="X"))
        h.add_edge(Edge("X", "out", "X", "in"))
        h.expose_input("X", "in", "x")
        h.expose_output("X", "out", "y")
        r = validate(h)
        assert r.valid
        assert len(r.warnings) >= 1
        assert any("num_loop_steps" in w for w in r.warnings)

    def test_cycle_with_num_loop_steps_zero_no_warning(self):
        """num_loop_steps=0 is a valid explicit value and should suppress warnings."""
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        h.add_edge(Edge("A", "out", "B", "in"))
        h.add_edge(Edge("B", "out", "A", "in"))
        h.expose_input("A", "in", "x")
        h.expose_output("B", "out", "y")
        h.metadata = {"num_loop_steps": 0}
        r = validate(h)
        assert r.valid
        assert len(r.warnings) == 0


class TestValidatorUnknownEdgeNode:
    def test_edge_with_unknown_source_node(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.expose_input("A", "in", "x")
        h.expose_output("A", "out", "y")
        h._edges.append(Edge("GHOST", "out", "A", "in"))
        r = validate(h)
        assert not r.valid
        assert any("GHOST" in e for e in r.errors)

    def test_edge_with_unknown_target_node(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.expose_input("A", "in", "x")
        h.expose_output("A", "out", "y")
        h._edges.append(Edge("A", "out", "GHOST", "in"))
        r = validate(h)
        assert not r.valid
        assert any("GHOST" in e for e in r.errors)


class TestValidatorEdgeTargetPortMissing:
    def test_unknown_target_port(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        h._edges.append(Edge("A", "out", "B", "nonexistent"))
        h._in_edges.setdefault("B", []).append(h._edges[-1])
        h._out_edges.setdefault("A", []).append(h._edges[-1])
        r = validate(h)
        assert not r.valid
        assert any("nonexistent" in e and "not found" in e for e in r.errors)


class TestValidatorWorkflowLevelPorts:
    def test_workflow_validate_with_edges(self):
        """Cover workflow-level _find_port that reads from get_output_spec/get_input_spec."""
        from yggdrasill.foundation.registry import BlockRegistry
        from yggdrasill.workflow.workflow import Workflow
        reg = BlockRegistry()
        reg.register("test/identity_task", IdentityTaskNode)
        hg1 = Hypergraph.from_config({
            "graph_id": "h1",
            "nodes": [{"node_id": "N", "block_type": "test/identity_task"}],
            "edges": [],
            "exposed_inputs": [{"node_id": "N", "port_name": "in", "name": "in"}],
            "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "out"}],
        }, registry=reg)
        hg2 = Hypergraph.from_config({
            "graph_id": "h2",
            "nodes": [{"node_id": "N", "block_type": "test/identity_task"}],
            "edges": [],
            "exposed_inputs": [{"node_id": "N", "port_name": "in", "name": "in"}],
            "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "out"}],
        }, registry=reg)
        w = Workflow()
        w.add_node("g1", hg1)
        w.add_node("g2", hg2)
        w.add_edge("g1", "out", "g2", "in")
        w.expose_input("g1", "in", "x")
        w.expose_output("g2", "out", "y")
        r = validate(w)
        assert r.valid


class TestValidatorWorkflowExposedOutput:
    def test_workflow_exposed_output_unknown_node(self):
        from yggdrasill.workflow.workflow import Workflow
        from yggdrasill.foundation.registry import BlockRegistry
        reg = BlockRegistry()
        reg.register("test/identity_task", IdentityTaskNode)
        hg = Hypergraph.from_config({
            "graph_id": "hg",
            "nodes": [{"node_id": "N", "block_type": "test/identity_task"}],
            "edges": [],
            "exposed_inputs": [{"node_id": "N", "port_name": "in", "name": "in"}],
            "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "out"}],
        }, registry=reg)
        w = Workflow()
        w.add_node("g1", hg)
        w.expose_input("g1", "in", "x")
        w.expose_output("g1", "out", "y")
        w._exposed_outputs.append({"graph_id": "GHOST", "port_name": "out"})
        r = validate(w)
        assert not r.valid
        assert any("GHOST" in e for e in r.errors)


class TestValidatorWorkflowCustomDtype:
    def test_workflow_custom_dtype_fallback(self):
        """Cover _find_port dtype fallback for non-PortType strings."""
        from yggdrasill.foundation.registry import BlockRegistry
        from yggdrasill.workflow.workflow import Workflow
        reg = BlockRegistry()
        reg.register("test/identity_task", IdentityTaskNode)
        hg1 = Hypergraph.from_config({
            "graph_id": "h1",
            "nodes": [{"node_id": "N", "block_type": "test/identity_task"}],
            "edges": [],
            "exposed_inputs": [{"node_id": "N", "port_name": "in", "name": "in"}],
            "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "out"}],
        }, registry=reg)
        hg2 = Hypergraph.from_config({
            "graph_id": "h2",
            "nodes": [{"node_id": "N", "block_type": "test/identity_task"}],
            "edges": [],
            "exposed_inputs": [{"node_id": "N", "port_name": "in", "name": "in"}],
            "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "out"}],
        }, registry=reg)
        orig_get_output_spec = hg1.get_output_spec
        def custom_output_spec(include_dtype=False):
            specs = orig_get_output_spec(include_dtype=include_dtype)
            if include_dtype:
                for s in specs:
                    s["dtype"] = "exotic_custom_type"
            return specs
        hg1.get_output_spec = custom_output_spec
        w = Workflow()
        w.add_node("g1", hg1)
        w.add_node("g2", hg2)
        w.add_edge("g1", "out", "g2", "in")
        w.expose_input("g1", "in", "x")
        w.expose_output("g2", "out", "y")
        r = validate(w)
        assert r.valid


class TestValidatorExposedPortDirection:
    def test_exposed_input_wrong_direction(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h._exposed_inputs.append({"node_id": "A", "port_name": "out"})
        r = validate(h)
        assert not r.valid
        assert any("out" in e and "not found as input" in e for e in r.errors)

    def test_exposed_output_wrong_direction(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h._exposed_outputs.append({"node_id": "A", "port_name": "in"})
        r = validate(h)
        assert not r.valid
        assert any("in" in e and "not found as output" in e for e in r.errors)
