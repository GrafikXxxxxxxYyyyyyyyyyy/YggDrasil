import pytest
from yggdrasill.foundation.registry import BlockRegistry
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.engine.planner import clear_plan_cache

import yggdrasill.task_nodes.stubs  # noqa: F401

ALL_STUB_TYPES = [
    "backbone/identity",
    "injector/identity",
    "conjector/identity",
    "inner_module/identity",
    "outer_module/identity",
    "helper/identity",
    "converter/identity",
]


class TestStubsRegistered:
    @pytest.mark.parametrize("block_type", ALL_STUB_TYPES)
    def test_registered(self, block_type):
        reg = BlockRegistry.global_registry()
        assert block_type in reg

    @pytest.mark.parametrize("block_type", ALL_STUB_TYPES)
    def test_build_and_run(self, block_type):
        reg = BlockRegistry.global_registry()
        node = reg.build({"block_type": block_type, "node_id": "N"})
        assert node.node_id == "N"
        assert node.block_type == block_type
        ports = node.declare_ports()
        in_names = [p.name for p in ports if p.is_input]
        out_names = [p.name for p in ports if p.is_output]
        assert len(in_names) >= 1
        assert len(out_names) >= 1
        test_in = {name: 42 for name in in_names}
        result = node.run(test_in)
        for out_name in out_names:
            assert out_name in result


class TestStubIntegration:
    def test_config_built_hypergraph(self):
        clear_plan_cache()
        reg = BlockRegistry.global_registry()
        cfg = {
            "graph_id": "stub_chain",
            "nodes": [
                {"node_id": "enc", "block_type": "converter/identity"},
                {"node_id": "bb", "block_type": "backbone/identity"},
                {"node_id": "dec", "block_type": "converter/identity"},
            ],
            "edges": [
                {"source_node": "enc", "source_port": "output", "target_node": "bb", "target_port": "latent"},
                {"source_node": "bb", "source_port": "pred", "target_node": "dec", "target_port": "input"},
            ],
            "exposed_inputs": [
                {"node_id": "enc", "port_name": "input", "name": "x"},
                {"node_id": "bb", "port_name": "timestep", "name": "timestep"},
            ],
            "exposed_outputs": [{"node_id": "dec", "port_name": "output", "name": "y"}],
        }
        h = Hypergraph.from_config(cfg, registry=reg)
        out = h.run({"x": "hello", "timestep": 0})
        assert out["y"] == "hello"

    def test_backbone_identity_node_attributes(self):
        reg = BlockRegistry.global_registry()
        node = reg.build({"block_type": "backbone/identity", "node_id": "B1", "block_id": "my_bb"})
        assert node.block_id == "my_bb"
        assert node.node_id == "B1"
        assert hasattr(node, "role")
        from yggdrasill.task_nodes.roles import Role
        assert node.role == Role.BACKBONE


class TestStubForwardContracts:
    def test_backbone_identity(self):
        reg = BlockRegistry.global_registry()
        node = reg.build({"block_type": "backbone/identity", "node_id": "B"})
        result = node.forward({"latent": "L", "timestep": "T"})
        assert result == {"pred": "L"}

    def test_injector_identity(self):
        reg = BlockRegistry.global_registry()
        node = reg.build({"block_type": "injector/identity", "node_id": "I"})
        result = node.forward({"condition": "C"})
        assert result == {"adapted": "C"}

    def test_conjector_identity(self):
        reg = BlockRegistry.global_registry()
        node = reg.build({"block_type": "conjector/identity", "node_id": "C"})
        result = node.forward({"input": "X"})
        assert result == {"condition": "X"}

    def test_inner_module_identity(self):
        reg = BlockRegistry.global_registry()
        node = reg.build({"block_type": "inner_module/identity", "node_id": "IM"})
        result = node.forward({"latent": "L", "timestep": "T", "pred": "P"})
        assert result == {"next_latent": "L", "next_timestep": "T"}

    def test_outer_module_identity(self):
        reg = BlockRegistry.global_registry()
        node = reg.build({"block_type": "outer_module/identity", "node_id": "OM"})
        result = node.forward({"input": "X"})
        assert result == {"output": "X"}

    def test_helper_identity(self):
        reg = BlockRegistry.global_registry()
        node = reg.build({"block_type": "helper/identity", "node_id": "H"})
        result = node.forward({"query": "Q"})
        assert result == {"result": "Q"}

    def test_converter_identity(self):
        reg = BlockRegistry.global_registry()
        node = reg.build({"block_type": "converter/identity", "node_id": "CV"})
        result = node.forward({"input": "X"})
        assert result == {"output": "X"}
