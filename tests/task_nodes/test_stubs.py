import pytest
from yggdrasill.foundation.registry import BlockRegistry
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.engine.planner import clear_plan_cache

import yggdrasill.task_nodes.stubs  # noqa: F401 -- triggers registration


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
    @pytest.fixture(autouse=True)
    def _reset(self):
        yield
        # don't reset global -- stubs should stay registered

    @pytest.mark.parametrize("block_type", ALL_STUB_TYPES)
    def test_registered(self, block_type):
        reg = BlockRegistry.global_registry()
        assert block_type in reg

    @pytest.mark.parametrize("block_type", ALL_STUB_TYPES)
    def test_build_and_run(self, block_type):
        reg = BlockRegistry.global_registry()
        node = reg.build({"block_type": block_type, "node_id": "N"})
        assert node.node_id == "N"  # type: ignore[attr-defined]
        assert node.block_type == block_type
        ports = node.declare_ports()  # type: ignore[attr-defined]
        in_names = [p.name for p in ports if p.is_input]
        out_names = [p.name for p in ports if p.is_output]
        assert len(in_names) >= 1
        assert len(out_names) >= 1
        test_in = {in_names[0]: 42}
        result = node.run(test_in)  # type: ignore[attr-defined]
        assert out_names[0] in result


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
                {"source_node": "enc", "source_port": "out", "target_node": "bb", "target_port": "in"},
                {"source_node": "bb", "source_port": "out", "target_node": "dec", "target_port": "in"},
            ],
            "exposed_inputs": [{"node_id": "enc", "port_name": "in", "name": "x"}],
            "exposed_outputs": [{"node_id": "dec", "port_name": "out", "name": "y"}],
        }
        h = Hypergraph.from_config(cfg, registry=reg)
        out = h.run({"x": "hello"})
        assert out["y"] == "hello"

    def test_backbone_identity_node_attributes(self):
        reg = BlockRegistry.global_registry()
        node = reg.build({"block_type": "backbone/identity", "node_id": "B1", "block_id": "my_bb"})
        assert node.block_id == "my_bb"
        assert node.node_id == "B1"  # type: ignore[attr-defined]
        assert hasattr(node, "role")
        from yggdrasill.task_nodes.roles import Role
        assert node.role == Role.BACKBONE  # type: ignore[attr-defined]
