"""Tests for foundation.Port."""

import pytest

from yggdrasill.foundation.port import (
    Port,
    PortDirection,
    PortAggregation,
    PortType,
)


def test_port_basic() -> None:
    p = Port("in1", PortDirection.IN, dtype=PortType.TENSOR)
    assert p.name == "in1"
    assert p.direction == PortDirection.IN
    assert p.dtype == PortType.TENSOR
    assert p.optional is False
    assert p.aggregation == PortAggregation.SINGLE
    assert p.is_input is True
    assert p.is_output is False


def test_port_empty_name_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Port("", PortDirection.IN)
    with pytest.raises(ValueError, match="non-empty"):
        Port("  ", PortDirection.OUT)


def test_port_output_aggregation_must_be_single() -> None:
    with pytest.raises(ValueError, match="SINGLE"):
        Port("out", PortDirection.OUT, aggregation=PortAggregation.CONCAT)


def test_port_compatible_with() -> None:
    out_any = Port("o", PortDirection.OUT, dtype=PortType.ANY)
    in_any = Port("i", PortDirection.IN, dtype=PortType.ANY)
    in_tensor = Port("i", PortDirection.IN, dtype=PortType.TENSOR)
    out_tensor = Port("o", PortDirection.OUT, dtype=PortType.TENSOR)
    assert out_any.compatible_with(in_any) is True
    assert out_any.compatible_with(in_tensor) is True
    assert out_tensor.compatible_with(in_tensor) is True
    assert out_tensor.compatible_with(in_any) is True
    assert in_any.compatible_with(out_any) is False  # wrong direction
    assert out_any.compatible_with(out_any) is False
