from __future__ import annotations

from typing import Any, Dict, List, Tuple

from yggdrasill.foundation.port import PortAggregation


class EdgeBuffers:
    """Per-run storage keyed by (node_id, port_name)."""

    def __init__(self) -> None:
        self._data: Dict[Tuple[str, str], Any] = {}
        self._multi: Dict[Tuple[str, str], List[Any]] = {}

    def write(self, node_id: str, port_name: str, value: Any) -> None:
        key = (node_id, port_name)
        self._data[key] = value

    def append(self, node_id: str, port_name: str, value: Any) -> None:
        """Append a value (for multi-edge aggregation)."""
        key = (node_id, port_name)
        self._multi.setdefault(key, []).append(value)

    def read(self, node_id: str, port_name: str) -> Any:
        return self._data.get((node_id, port_name))

    def has(self, node_id: str, port_name: str) -> bool:
        return (node_id, port_name) in self._data

    def aggregate(
        self,
        node_id: str,
        port_name: str,
        policy: PortAggregation = PortAggregation.SINGLE,
    ) -> Any:
        key = (node_id, port_name)
        multi = self._multi.get(key, [])
        if not multi:
            return self._data.get(key)
        if policy == PortAggregation.FIRST:
            return multi[0]
        if policy == PortAggregation.CONCAT:
            return multi
        if policy == PortAggregation.SUM:
            return sum(multi)
        # SINGLE or DICT -- just return the last written
        return multi[-1] if multi else self._data.get(key)
