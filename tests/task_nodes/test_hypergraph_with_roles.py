"""PHASE_4 §12: Hypergraph from config with block_type=backbone/identity etc."""
from __future__ import annotations

import pytest

from yggdrasill.engine.planner import clear_plan_cache
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.foundation.registry import BlockRegistry

import yggdrasill.task_nodes.stubs  # noqa: F401


class TestHypergraphWithRoles:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        clear_plan_cache()

    def test_converter_backbone_chain(self):
        cfg = {
            "graph_id": "roles_chain",
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
                {"node_id": "bb", "port_name": "timestep", "name": "t"},
            ],
            "exposed_outputs": [{"node_id": "dec", "port_name": "output", "name": "y"}],
        }
        reg = BlockRegistry.global_registry()
        h = Hypergraph.from_config(cfg, registry=reg)
        out = h.run({"x": "data", "t": 0})
        assert "y" in out

    def test_backbone_inner_module_cycle(self):
        cfg = {
            "graph_id": "cycle_roles",
            "metadata": {"num_loop_steps": 3},
            "nodes": [
                {"node_id": "bb", "block_type": "backbone/identity"},
                {"node_id": "im", "block_type": "inner_module/identity"},
            ],
            "edges": [
                {"source_node": "bb", "source_port": "pred", "target_node": "im", "target_port": "pred"},
                {"source_node": "im", "source_port": "next_latent", "target_node": "bb", "target_port": "latent"},
                {"source_node": "im", "source_port": "next_timestep", "target_node": "bb", "target_port": "timestep"},
            ],
            "exposed_inputs": [
                {"node_id": "bb", "port_name": "latent", "name": "latent"},
                {"node_id": "bb", "port_name": "timestep", "name": "timestep"},
                {"node_id": "im", "port_name": "latent", "name": "im_latent"},
                {"node_id": "im", "port_name": "timestep", "name": "im_timestep"},
            ],
            "exposed_outputs": [
                {"node_id": "im", "port_name": "next_latent", "name": "out_latent"},
            ],
        }
        reg = BlockRegistry.global_registry()
        h = Hypergraph.from_config(cfg, registry=reg, validate=False)
        out = h.run(
            {"latent": "lat", "timestep": 0, "im_latent": "lat", "im_timestep": 0},
            validate_before=False,
        )
        assert "out_latent" in out

    def test_conjector_backbone(self):
        cfg = {
            "graph_id": "cond",
            "nodes": [
                {"node_id": "cj", "block_type": "conjector/identity"},
                {"node_id": "bb", "block_type": "backbone/identity"},
            ],
            "edges": [
                {"source_node": "cj", "source_port": "condition", "target_node": "bb", "target_port": "condition"},
            ],
            "exposed_inputs": [
                {"node_id": "cj", "port_name": "input", "name": "prompt"},
                {"node_id": "bb", "port_name": "latent", "name": "latent"},
                {"node_id": "bb", "port_name": "timestep", "name": "t"},
            ],
            "exposed_outputs": [
                {"node_id": "bb", "port_name": "pred", "name": "pred"},
            ],
        }
        reg = BlockRegistry.global_registry()
        h = Hypergraph.from_config(cfg, registry=reg)
        out = h.run({"prompt": "hello", "latent": "z", "t": 1})
        assert "pred" in out

    def test_all_seven_stubs_registered(self):
        reg = BlockRegistry.global_registry()
        for role in [
            "backbone/identity", "injector/identity", "conjector/identity",
            "inner_module/identity", "outer_module/identity",
            "helper/identity", "converter/identity",
        ]:
            assert role in reg, f"{role} not registered"
