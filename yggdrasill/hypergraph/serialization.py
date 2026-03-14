"""Config + Checkpoint serialization for blocks and hypergraphs.

Each level is saved as:
  <dir>/config.json    -- structure/metadata (JSON)
  <dir>/checkpoint.pkl -- dynamic state/weights (pickle)

block_id deduplication: when saving a hypergraph, blocks with the same
block_id share a single checkpoint entry.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from yggdrasill.engine.structure import Hypergraph
from yggdrasill.foundation.block import AbstractBaseBlock


SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def save_config(config: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Block-level save / load
# ---------------------------------------------------------------------------

def save_block(block: AbstractBaseBlock, directory: str | Path) -> Path:
    """Save a block to <directory>/config.json + checkpoint.pkl."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    config = block.get_config()
    config["schema_version"] = SCHEMA_VERSION
    save_config(config, directory / "config.json")

    state = block.state_dict()
    if state:
        save_checkpoint(state, directory / "checkpoint.pkl")

    return directory


def load_block(
    directory: str | Path,
    *,
    registry: Optional[Any] = None,
) -> AbstractBaseBlock:
    """Load a block from <directory>/config.json + checkpoint.pkl."""
    from yggdrasill.foundation.registry import BlockRegistry

    directory = Path(directory)
    config = load_config(directory / "config.json")

    reg = registry or BlockRegistry.global_registry()
    block = reg.build(config)

    ckpt_path = directory / "checkpoint.pkl"
    if ckpt_path.exists():
        state = load_checkpoint(ckpt_path)
        block.load_state_dict(state, strict=False)

    return block


# ---------------------------------------------------------------------------
# Hypergraph-level save / load
# ---------------------------------------------------------------------------

def save_hypergraph(hypergraph: Hypergraph, directory: str | Path) -> Path:
    """Save <directory>/config.json + checkpoint.pkl (with block_id dedup)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    cfg = hypergraph.to_config()
    cfg["schema_version"] = SCHEMA_VERSION
    save_config(cfg, directory / "config.json")

    full_state = hypergraph.state_dict()
    if full_state:
        deduped = _deduplicate_state(hypergraph, full_state)
        save_checkpoint(deduped, directory / "checkpoint.pkl")

    return directory


def load_hypergraph(
    directory: str | Path,
    *,
    registry: Optional[Any] = None,
) -> Hypergraph:
    """Load <directory>/config.json + checkpoint.pkl, rebuild via registry."""
    from yggdrasill.foundation.registry import BlockRegistry

    directory = Path(directory)
    config = load_config(directory / "config.json")
    reg = registry or BlockRegistry.global_registry()
    h = Hypergraph.from_config(config, registry=reg)

    ckpt_path = directory / "checkpoint.pkl"
    if ckpt_path.exists():
        state = load_checkpoint(ckpt_path)
        expanded = _expand_deduped_state(h, state)
        h.load_state_dict(expanded, strict=False)

    return h


# ---------------------------------------------------------------------------
# block_id deduplication
# ---------------------------------------------------------------------------

def _deduplicate_state(
    hypergraph: Hypergraph, state: Dict[str, Any],
) -> Dict[str, Any]:
    """Collapse nodes with the same block_id into one checkpoint entry."""
    seen_block_ids: Dict[str, str] = {}  # block_id -> first node_id
    aliases: Dict[str, str] = {}  # node_id -> alias (first node_id with same block_id)
    deduped: Dict[str, Any] = {}

    for nid in sorted(state.keys()):
        node = hypergraph.get_node(nid)
        block_id = getattr(node, "block_id", nid) if node is not None else nid
        if block_id in seen_block_ids:
            aliases[nid] = seen_block_ids[block_id]
        else:
            seen_block_ids[block_id] = nid
            deduped[nid] = state[nid]

    if aliases:
        deduped["_aliases"] = aliases

    return deduped


def _expand_deduped_state(
    hypergraph: Hypergraph, state: Dict[str, Any],
) -> Dict[str, Any]:
    """Expand aliases back to per-node state dicts."""
    aliases: Dict[str, str] = state.pop("_aliases", {})
    expanded = dict(state)

    for nid, alias_target in aliases.items():
        if alias_target in expanded:
            expanded[nid] = expanded[alias_target]

    return expanded
