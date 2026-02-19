"""
Abstract Base Block: storage + execution + identity.

Canon: WorldGenerator_2.0/Abstract_Block_And_Node.md §2, TODO_01 §1.
- declare_ports(), forward(inputs) -> outputs, block_type, block_id
- state_dict(), load_state_dict(), trainable_parameters()
- Optional: train/eval mode, freeze/unfreeze.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional

from yggdrasill.foundation.port import Port


def _state_dict_with_sub_blocks(
    self: AbstractBaseBlock,
    own_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge own_state with state_dict of all sub-blocks (prefix by name)."""
    out = dict(own_state)
    for sub_name, sub_block in self.get_sub_blocks().items():
        for k, v in sub_block.state_dict().items():
            out[f"{sub_name}.{k}"] = v
    return out


def _load_state_dict_into_sub_blocks(
    self: AbstractBaseBlock,
    state: Dict[str, Any],
    strict: bool,
) -> Dict[str, Any]:
    """Dispatch prefixed keys to sub-blocks; return remaining state for self."""
    subs = self.get_sub_blocks()
    rest = {}
    for key, value in state.items():
        if "." in key:
            prefix, _ = key.split(".", 1)
            if prefix in subs:
                continue  # will be passed to sub-block
        rest[key] = value
    for sub_name, sub_block in subs.items():
        sub_state = {
            k[len(sub_name) + 1 :]: v
            for k, v in state.items()
            if k.startswith(sub_name + ".")
        }
        if sub_state:
            sub_block.load_state_dict(sub_state, strict=strict)
    return rest


class AbstractBaseBlock(ABC):
    """
    Minimal unit: stores (config, tensors, state) and executes (inputs → outputs).

    Identity: block_type (kind of block), block_id (instance id in scope).
    Contract: declare_ports(), forward(inputs) -> outputs, state_dict/load_state_dict.
    """

    def __init__(
        self,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._block_id = block_id or self._default_block_id()
        self._config = dict(config or {})
        self._training = True
        self._frozen = False

    def _default_block_id(self) -> str:
        """Override to provide default block_id; default uses class name + id(self)."""
        return f"{type(self).__name__}_{id(self)}"

    # --- Identity ---

    @property
    def block_type(self) -> str:
        """Type identifier for registry and config. Override in subclass."""
        return type(self).__name__

    @property
    def block_id(self) -> str:
        return self._block_id

    @property
    def config(self) -> Dict[str, Any]:
        """Copy of config used to build this block (for serialization)."""
        return dict(self._config)

    # --- Ports ---

    @abstractmethod
    def declare_ports(self) -> List[Port]:
        """Declare input/output ports (names, types, optional, aggregation)."""
        ...

    def get_input_ports(self) -> List[Port]:
        return [p for p in self.declare_ports() if p.is_input]

    def get_output_ports(self) -> List[Port]:
        return [p for p in self.declare_ports() if p.is_output]

    def get_port(self, name: str) -> Optional[Port]:
        for p in self.declare_ports():
            if p.name == name:
                return p
        return None

    def get_sub_blocks(self) -> Dict[str, "AbstractBaseBlock"]:
        """
        Return named sub-blocks for composite serialization (TODO_01 §1.3).
        Override to return a dict name -> block; default is empty.
        state_dict/load_state_dict merge sub-block state with prefix "name.".
        """
        return {}

    # --- Execution ---

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute: inputs dict (port name -> value) -> outputs dict (port name -> value).
        Must match declared ports.
        """
        ...

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for forward (canon allows forward/run)."""
        return self.forward(inputs)

    # --- Train / Eval ---

    @property
    def training(self) -> bool:
        return self._training

    def train(self, mode: bool = True) -> AbstractBaseBlock:
        self._training = mode
        return self

    def eval(self) -> AbstractBaseBlock:
        return self.train(False)

    # --- Freeze / Unfreeze ---

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> AbstractBaseBlock:
        self._frozen = True
        return self

    def unfreeze(self) -> AbstractBaseBlock:
        self._frozen = False
        return self

    # --- State (checkpoint) ---

    def state_dict(self) -> Dict[str, Any]:
        """
        All tensors/parameters needed to reproduce the block.
        Override to add module-specific state; then call
        return _state_dict_with_sub_blocks(self, out) to include sub-blocks.
        """
        return _state_dict_with_sub_blocks(self, {})

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        """
        Restore state from checkpoint. Override for custom loading.
        Sub-blocks (get_sub_blocks()) receive prefixed keys automatically.
        strict: if True, raise on unexpected or missing keys.
        """
        rest = _load_state_dict_into_sub_blocks(self, state, strict)
        if strict and rest:
            raise KeyError(f"Unexpected keys in state_dict: {list(rest)}")

    def trainable_parameters(self) -> Iterator[Any]:
        """
        Parameters to pass to an optimizer. Default: empty.
        Override in subclasses (e.g. return model.parameters() for PyTorch).
        """
        return
        yield  # make this a generator

    def __repr__(self) -> str:
        return f"{type(self).__name__}(block_id={self.block_id!r}, block_type={self.block_type!r})"
