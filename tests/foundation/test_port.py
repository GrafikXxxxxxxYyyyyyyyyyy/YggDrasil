import pytest

from yggdrasill.foundation.port import (
    Port,
    PortAggregation,
    PortDirection,
    PortType,
)


class TestPortCreation:
    def test_basic_creation(self):
        p = Port("latent", PortDirection.IN, PortType.TENSOR)
        assert p.name == "latent"
        assert p.direction == PortDirection.IN
        assert p.dtype == PortType.TENSOR
        assert p.optional is False
        assert p.aggregation == PortAggregation.SINGLE

    def test_defaults(self):
        p = Port("x", PortDirection.OUT)
        assert p.dtype == PortType.ANY
        assert p.optional is False
        assert p.aggregation == PortAggregation.SINGLE

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            Port("", PortDirection.IN)

    def test_whitespace_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            Port("   ", PortDirection.OUT)

    def test_output_port_with_non_single_aggregation_raises(self):
        with pytest.raises(ValueError, match="aggregation=SINGLE"):
            Port("out", PortDirection.OUT, aggregation=PortAggregation.CONCAT)

    def test_input_port_concat_aggregation(self):
        p = Port("in", PortDirection.IN, aggregation=PortAggregation.CONCAT)
        assert p.aggregation == PortAggregation.CONCAT

    def test_input_port_first_aggregation(self):
        p = Port("in", PortDirection.IN, aggregation=PortAggregation.FIRST)
        assert p.aggregation == PortAggregation.FIRST

    def test_frozen(self):
        p = Port("x", PortDirection.IN)
        with pytest.raises(AttributeError):
            p.name = "y"  # type: ignore[misc]


class TestPortProperties:
    def test_is_input(self):
        p = Port("x", PortDirection.IN)
        assert p.is_input is True
        assert p.is_output is False

    def test_is_output(self):
        p = Port("x", PortDirection.OUT)
        assert p.is_input is False
        assert p.is_output is True


class TestPortCompatibility:
    def test_out_to_in_same_type(self):
        src = Port("a", PortDirection.OUT, PortType.TENSOR)
        dst = Port("b", PortDirection.IN, PortType.TENSOR)
        assert src.compatible_with(dst) is True

    def test_out_any_to_in_tensor(self):
        src = Port("a", PortDirection.OUT, PortType.ANY)
        dst = Port("b", PortDirection.IN, PortType.TENSOR)
        assert src.compatible_with(dst) is True

    def test_out_tensor_to_in_any(self):
        src = Port("a", PortDirection.OUT, PortType.TENSOR)
        dst = Port("b", PortDirection.IN, PortType.ANY)
        assert src.compatible_with(dst) is True

    def test_incompatible_types(self):
        src = Port("a", PortDirection.OUT, PortType.TENSOR)
        dst = Port("b", PortDirection.IN, PortType.TEXT)
        assert src.compatible_with(dst) is False

    def test_in_to_out_is_false(self):
        a = Port("a", PortDirection.IN, PortType.ANY)
        b = Port("b", PortDirection.OUT, PortType.ANY)
        assert a.compatible_with(b) is False

    def test_out_to_out_is_false(self):
        a = Port("a", PortDirection.OUT, PortType.ANY)
        b = Port("b", PortDirection.OUT, PortType.ANY)
        assert a.compatible_with(b) is False
