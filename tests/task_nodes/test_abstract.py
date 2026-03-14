import pytest
from yggdrasill.foundation.port import PortDirection
from yggdrasill.task_nodes.abstract import (
    AbstractBackbone,
    AbstractConjector,
    AbstractConverter,
    AbstractHelper,
    AbstractInjector,
    AbstractInnerModule,
    AbstractOuterModule,
)
from yggdrasill.task_nodes.roles import Role


class TestAbstractInstantiation:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            AbstractBackbone(node_id="X")

    def test_cannot_instantiate_abstract_injector(self):
        with pytest.raises(TypeError):
            AbstractInjector(node_id="X")


class TestAbstractRoles:
    def test_backbone_role(self):
        assert AbstractBackbone._role == Role.BACKBONE

    def test_injector_role(self):
        assert AbstractInjector._role == Role.INJECTOR

    def test_conjector_role(self):
        assert AbstractConjector._role == Role.CONJECTOR

    def test_inner_module_role(self):
        assert AbstractInnerModule._role == Role.INNER_MODULE

    def test_outer_module_role(self):
        assert AbstractOuterModule._role == Role.OUTER_MODULE

    def test_helper_role(self):
        assert AbstractHelper._role == Role.HELPER

    def test_converter_role(self):
        assert AbstractConverter._role == Role.CONVERTER


class TestCanonicalPorts:
    """Verify that each abstraction declares the correct canonical ports."""

    def _port_info(self, cls):
        """Instantiate a concrete stub to get ports. We use the stubs module."""
        from yggdrasill.task_nodes import stubs
        stub_map = {
            AbstractBackbone: stubs.IdentityBackbone,
            AbstractInjector: stubs.IdentityInjector,
            AbstractConjector: stubs.IdentityConjector,
            AbstractInnerModule: stubs.IdentityInnerModule,
            AbstractOuterModule: stubs.IdentityOuterModule,
            AbstractHelper: stubs.IdentityHelper,
            AbstractConverter: stubs.IdentityConverter,
        }
        instance = stub_map[cls](node_id="test")
        return instance.declare_ports()

    def test_backbone_ports(self):
        ports = self._port_info(AbstractBackbone)
        names = {p.name: p for p in ports}
        assert "latent" in names
        assert "timestep" in names
        assert "pred" in names
        assert names["latent"].direction == PortDirection.IN
        assert names["timestep"].direction == PortDirection.IN
        assert names["pred"].direction == PortDirection.OUT
        assert "condition" in names
        assert names["condition"].optional is True

    def test_injector_ports(self):
        ports = self._port_info(AbstractInjector)
        names = {p.name: p for p in ports}
        assert "condition" in names and names["condition"].direction == PortDirection.IN
        assert "adapted" in names and names["adapted"].direction == PortDirection.OUT

    def test_conjector_ports(self):
        ports = self._port_info(AbstractConjector)
        names = {p.name: p for p in ports}
        assert "input" in names and names["input"].direction == PortDirection.IN
        assert "condition" in names and names["condition"].direction == PortDirection.OUT

    def test_inner_module_ports(self):
        ports = self._port_info(AbstractInnerModule)
        names = {p.name: p for p in ports}
        assert "latent" in names and names["latent"].direction == PortDirection.IN
        assert "timestep" in names and names["timestep"].direction == PortDirection.IN
        assert "pred" in names and names["pred"].direction == PortDirection.IN
        assert "next_latent" in names and names["next_latent"].direction == PortDirection.OUT

    def test_outer_module_ports(self):
        ports = self._port_info(AbstractOuterModule)
        names = {p.name: p for p in ports}
        assert "input" in names and names["input"].optional is True
        assert "output" in names and names["output"].direction == PortDirection.OUT

    def test_helper_ports(self):
        ports = self._port_info(AbstractHelper)
        names = {p.name: p for p in ports}
        assert "query" in names and names["query"].direction == PortDirection.IN
        assert "result" in names and names["result"].direction == PortDirection.OUT

    def test_converter_ports(self):
        ports = self._port_info(AbstractConverter)
        names = {p.name: p for p in ports}
        assert "input" in names and names["input"].direction == PortDirection.IN
        assert "output" in names and names["output"].direction == PortDirection.OUT
