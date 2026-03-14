"""Textual Inversion (embeddings) support."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractHelper


class TextualInversionNode(AbstractHelper):
    """Loads textual inversion embeddings into tokenizer + text encoder.

    This node augments the prompt vocabulary before encoding.
    It should run before the prompt encoder node in the graph.
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        pipe: Any = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)
        self._pipe = pipe

    @property
    def block_type(self) -> str:
        return "adapter/textual_inversion"

    def declare_ports(self) -> List[Port]:
        return [
            Port("trigger", PortDirection.IN, PortType.ANY, optional=True),
            Port("result", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        embeddings = self._config.get("embeddings", [])
        loaded_tokens: List[str] = []

        for emb in embeddings:
            path = emb.get("path", "")
            token = emb.get("token")

            kwargs: Dict[str, Any] = {}
            if token:
                kwargs["token"] = token

            self._pipe.load_textual_inversion(path, **kwargs)

            loaded_tokens.append(token or path.split("/")[-1])

        return {"result": {"loaded_tokens": loaded_tokens}}
