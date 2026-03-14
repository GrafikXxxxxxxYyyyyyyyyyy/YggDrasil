import pytest
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
            AbstractBackbone(node_id="X")  # type: ignore[abstract]

    def test_cannot_instantiate_abstract_injector(self):
        with pytest.raises(TypeError):
            AbstractInjector(node_id="X")  # type: ignore[abstract]


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
