# DONE 01: Foundation (Fundamental Level)

**Project canon:** [CANON.md](../WorldGenerator_2.0/CANON.md). **Canon:** [TODO_01_FOUNDATION.md](../WorldGenerator_2.0/TODO_01_FOUNDATION.md), [Abstract_Block_And_Node.md](../WorldGenerator_2.0/Abstract_Block_And_Node.md), [AGENT_SYSTEMS_SUPPORT.md](../WorldGenerator_2.0/AGENT_SYSTEMS_SUPPORT.md) (flexibility, agent block contract).

This document describes what was implemented for the foundation level, how it works, where it lives in the codebase, and how it fits into the future architecture.

---

## 1. What Was Done

- **Abstract Base Block** — base class with `declare_ports()`, `forward(inputs) -> outputs`, `block_type`, `block_id`, `state_dict()`, `load_state_dict()`, `trainable_parameters()`, and optional `train`/`eval`, `freeze`/`unfreeze`.
- **Ports** — dataclass `Port(name, direction, dtype, optional, aggregation)` with direction IN/OUT, type (tensor, dict, any, image, text, …), optional flag, and aggregation (single, concat, sum, first, dict). Compatibility check: `source_port.compatible_with(target_port)` for edges.
- **Abstract Graph Node** — `Node(node_id, block)`; node holds only identity and block reference; edges live in the graph.
- **Edge** — immutable `Edge(source_node, source_port, target_node, target_port)`.
- **Graph** — container of nodes (node_id → Node), edges (with O(1) in/out indexes), and **exposed inputs/outputs** (contract for executor and pipeline); `add_node`, `add_edge`, `expose_input(node_id, port_name, name?)`, `expose_output(node_id, port_name, name?)`; `get_input_spec()` / `get_output_spec()`; `to_config()` (includes `schema_version`, `exposed_inputs`, `exposed_outputs`) / `from_config()`; `state_dict()` / `load_state_dict()` and `load_from_checkpoint(...)`.
- **Block Registry** — `BlockRegistry.register(block_type, class)` and `build(config)` creating a block from config (requires `block_type` or `type`); optional `@register_block("name")` decorator. Global registry: `BlockRegistry.global_registry()`.

All of the above is implemented in code, covered by tests, and ready for use by Task Nodes (TODO_02) and Graph Engine (TODO_03).

---

## 2. How It Is Implemented in Code

### 2.1 Directory and module layout

```
yggdrasill/
  __init__.py           # re-exports foundation public API + __version__
  foundation/
    __init__.py         # re-exports Port, Block, Node, Edge, Graph, Registry
    port.py             # Port, PortDirection, PortAggregation, PortType
    block.py            # AbstractBaseBlock
    node.py             # Node
    graph.py            # Edge, Graph
    registry.py         # BlockRegistry, register_block

tests/
  foundation/
    helpers.py          # AddBlock, IdentityBlock (concrete blocks for tests)
    test_port.py
    test_block.py
    test_node.py
    test_registry.py
    test_graph.py
```

### 2.2 Ports

- **Port** — frozen dataclass: `name`, `direction` (IN/OUT), `dtype` (PortType), `optional`, `aggregation`. Output ports must use `PortAggregation.SINGLE`. Empty name raises. `compatible_with(other)` checks direction and type (ANY matches any).
- **PortType** — enum: TENSOR, DICT, ANY, IMAGE, TEXT, AUDIO, VIDEO.

### 2.3 Abstract Base Block

- **AbstractBaseBlock(block_id=None, *, config=None)** — base constructor; subclasses override `block_type`, `declare_ports()`, `forward(inputs)`.
- **Identity:** `block_type` (default: class name), `block_id` (default from `_default_block_id()`), `config` (copy of construction config).
- **Execution:** `forward(inputs: Dict[str, Any]) -> Dict[str, Any]`; `run()` is an alias.
- **State:** `state_dict()` returns a dict (default empty); `load_state_dict(state, strict=True)` (default no-op in base).
- **Trainability:** `trainable_parameters()` returns an empty iterator by default; `train(mode)` / `eval()`, `freeze()` / `unfreeze()` with `training` and `frozen` properties.

- **Sub-blocks (composition):** `get_sub_blocks() -> Dict[str, AbstractBaseBlock]` (default `{}`). Base `state_dict()` merges sub-block state with key prefix `"{name}.{key}"`; `load_state_dict()` dispatches prefixed keys to the corresponding sub-block. Override `state_dict()` to add own state and call `return _state_dict_with_sub_blocks(self, out)` to include sub-blocks.

### 2.4 Node and Edge

- **Node(node_id, block)** — validates non-empty node_id; exposes `block`, `get_input_ports()`, `get_output_ports()` from the block.
- **Edge(source_node, source_port, target_node, target_port)** — frozen; all four fields must be non-empty.

### 2.5 Graph

- **Graph(graph_id=None)** — `add_node(node)`, `add_edge(edge)`. Edge validation: both nodes exist, both ports exist on the blocks, and source port is compatible with target port. **Exposed contract:** `expose_input(node_id, port_name, name=None)` and `expose_output(node_id, port_name, name=None)` mark graph-level inputs/outputs for the executor (TODO_03) and pipeline; `get_input_spec()` / `get_output_spec()` return lists of `{node_id, port_name, name?}`.
- **Edge indexes:** `_in_edges_by_node` and `_out_edges_by_node` give O(1) lookup for `get_edges_in(node_id)` / `get_edges_out(node_id)`.
- **Serialization:** `to_config()` → dict with `schema_version` (constant `GRAPH_CONFIG_SCHEMA_VERSION`), `graph_id`, `nodes`, `edges`, `exposed_inputs`, `exposed_outputs`. `Graph.from_config(config, registry)` builds blocks via registry, recreates nodes and edges, and restores exposed inputs/outputs.
- **Weights:** `state_dict()` returns `{node_id: block.state_dict() for each node}`; `load_state_dict(state, strict)` loads into each block by node_id. `load_from_checkpoint(config=..., checkpoint=..., ...)` rebuilds structure (including exposed ports) and loads weights.

### 2.6 Block Registry

- **BlockRegistry()** — `register(block_type, block_class_or_factory)`, `build(config)`. Config must have `block_type` or `type`; the rest (including `block_id`) is passed as `config` to the block constructor: `builder(block_id=block_id, config=rest)`.
- **register_block(block_type, registry=None)** — decorator to register a class; uses global registry if none given.

---

## 3. Examples

### 3.1 Define and run a block

```python
from yggdrasill.foundation import AbstractBaseBlock, Port, PortDirection, PortType

class MyBlock(AbstractBaseBlock):
    @property
    def block_type(self):
        return "my_block"

    def declare_ports(self):
        return [
            Port("x", PortDirection.IN, dtype=PortType.ANY),
            Port("y", PortDirection.OUT, dtype=PortType.ANY),
        ]

    def forward(self, inputs):
        return {"y": inputs.get("x", 0) * 2}

block = MyBlock(block_id="b1", config={"factor": 2})
out = block.forward({"x": 5})  # {"y": 10}
```

### 3.2 Graph and registry

```python
from yggdrasill.foundation import Node, Edge, Graph, BlockRegistry

reg = BlockRegistry()
reg.register("my_block", MyBlock)

g = Graph("g1")
g.add_node(Node("A", MyBlock(block_id="a1", config={})))
g.add_node(Node("B", MyBlock(block_id="b1", config={})))
g.add_edge(Edge("A", "y", "B", "x"))

cfg = g.to_config()
g2 = Graph.from_config(cfg, registry=reg)
ckpt = g.state_dict()
g2.load_state_dict(ckpt)
```

### 3.3 Load from checkpoint (structure + weights)

```python
g_new = Graph()
g_new.load_from_checkpoint(config=cfg, checkpoint=ckpt, registry=reg)
```

---

## 4. Tests

- **tests/foundation/test_port.py** — port creation, empty name, output aggregation, compatibility.
- **tests/foundation/test_block.py** — AddBlock/IdentityBlock: ports, forward, config, state_dict/load_state_dict, train/eval, freeze/unfreeze, run alias.
- **tests/foundation/test_node.py** — node creation, empty node_id.
- **tests/foundation/test_registry.py** — build from config (block_type/type), unknown/missing type, decorator registration.
- **tests/foundation/test_graph.py** — add node/edge, duplicate node, unknown node/port, to_config/from_config roundtrip, state_dict/load_state_dict, load_from_checkpoint with config + checkpoint.

Run: `python3 -m pytest tests/foundation -v`.

---

## 5. How This Fits the Architecture and Future Plans

- **Task nodes (TODO_02)** — Will subclass `AbstractBaseBlock`, implement concrete ports and `forward`, and register in the same `BlockRegistry`. No change to the foundation contract.
- **Graph engine (TODO_03)** — Will use this Graph (nodes + edges), add an executor: topological sort, feed inputs by edges, call `block.forward`, collect outputs. Foundation already provides structure and validation.
- **Pipeline / Stage / World (TODO_04–06)** — Will aggregate graphs or sub-graphs; their “weights” will ultimately be the state_dict of blocks inside; checkpoint load will rely on block/graph `state_dict`/`load_state_dict` and `load_from_checkpoint`.
- **Serialization (SERIALIZATION_AT_ALL_LEVELS)** — Block and graph config + state_dict are the base; upper levels will add their own config keys and possibly nest graph configs.
- **Trainability (TRAINABILITY_AT_ALL_LEVELS)** — `trainable_parameters()`, `freeze()`/`unfreeze()`, and state_dict on the block are the contract; graph-level training will iterate blocks and optionally freeze/unfreeze by node.

---

## 6. Short Guide

1. **Implement a new block:** Subclass `AbstractBaseBlock`, implement `block_type`, `declare_ports()`, `forward()`. Optionally override `state_dict`/`load_state_dict` and `trainable_parameters()`.
2. **Use in a graph:** Create a `Node(node_id, block)` and add it to a `Graph`; connect with `Edge(source_node, source_port, target_node, target_port)`. Use the same port names as in `declare_ports()` and ensure types are compatible.
3. **Save/load:** `graph.to_config()` for structure; `graph.state_dict()` for weights. Rebuild with `Graph.from_config(cfg, registry)` then `graph.load_state_dict(ckpt)`, or use `load_from_checkpoint(config=..., checkpoint=..., registry=...)`.
4. **Register blocks:** `BlockRegistry.global_registry().register("type_name", MyBlock)` or `@register_block("type_name")` on the class so that `Graph.from_config(..., registry=reg)` can build blocks from config.

---

## 7. Acceptance Criteria (from TODO_01)

- [x] Any block can be created from config (via registry), executed with inputs by port names, and produce outputs by port names.
- [x] Any block can be placed in a graph node; edges connect ports; graph structure (nodes + edges) is serialized and restored from config.
- [x] Block weights are serialized (state_dict) and loaded from checkpoint; at graph level, all node states are serialized and the graph can be loaded from checkpoint (config + weights).
- [x] Ports have names and types; adding an edge checks type compatibility; execution (when implemented) will pass data along edges to block ports.
- [x] Block may contain sub-blocks (same external contract; serialization/load for composite blocks): `get_sub_blocks()` returns named sub-blocks; `state_dict()`/`load_state_dict()` merge sub-block state with prefix `name.` (see §2.3 below).
- [x] Block registry allows registering classes and creating blocks by `block_type` from config; graph load from config uses the registry to create blocks in nodes.

Foundation is ready for the next level (task nodes and graph engine).

---

## 8. Что ещё не сделано на этом этапе

- **Режим «только API» для блока:** конфиг блока пока не предусматривает `backend: "api"`, `provider`, `model_id`; блоки с нулевыми локальными весами (LLM через API) не выделены в контракте. См. [LLM_API_SUPPORT.md](../WorldGenerator_2.0/LLM_API_SUPPORT.md).
- **Повторное использование одной модели:** при сборке графа/пайплайна один и тот же checkpoint_ref (или model_id для API) может приводить к нескольким экземплярам блока; пула «ref → один экземпляр» на уровне фундамента/реестра нет. См. [MODEL_REUSE.md](../WorldGenerator_2.0/MODEL_REUSE.md).
- **Агентный контракт блока:** расширение контракта для агента (состояние между вызовами, tool_calls/tool_results) и сохранение состояния агента в state_dict не реализованы. См. [AGENT_SYSTEMS_SUPPORT.md](../WorldGenerator_2.0/AGENT_SYSTEMS_SUPPORT.md).

---

## 9. Что нужно реализовать (по TODO_01 и канону)

- В конфиге блока: поддержка `backend: "api"`, `provider`, `model_id`; единый контракт (порты, forward, state_dict) для локальных и API-блоков.
- При сборке графа/пайплайна: идентификация «одна и та же модель» по checkpoint_ref/model_id и создание/возврат одного экземпляра блока на один ref (реестр или пул на уровне реестра/графа). При обучении с повторным использованием и при API — см. [TRAINING_REUSE_AND_API_SCENARIOS.md](../WorldGenerator_2.0/TRAINING_REUSE_AND_API_SCENARIOS.md).

Сводка по этапам: [СТАТУС_ПО_ЭТАПАМ.md](СТАТУС_ПО_ЭТАПАМ.md).

---

### Outline for further TODOs (WorldGenerator_2.0)

- **TODO_02** — Task nodes: Abstract Backbone, Solver, Codec, Conditioner, Tokenizer, Adapter, Guidance (double inheritance Block + role); port contracts; role rules table for auto-connect; AddNode(block_or_type, auto_connect=True), from_config/from_template.
- **TODO_03** — Graph engine: executor (topological order, SCC/cycles, buffer, `run(graph, inputs, training, num_loop_steps)`); validation; save_config/save_checkpoint, load; get_input_spec/get_output_spec (foundation already has exposed_inputs/outputs and spec).
- **TODO_04** — Pipeline: graph of graphs; nodes = Graph; edges connect external ports of graphs.
- **TODO_05** — Stage: graph of pipelines; state contract; execution conditions (Scheme).
- **TODO_06** — World: graph of stages; cycle; state schema; storage.
- **TODO_07** — Future: VLM, video, audio, streaming, backends, async, tracing.

### 7.1 Composite blocks (sub-blocks)

- **get_sub_blocks()** — override to return `Dict[str, AbstractBaseBlock]`. Default: `{}`.
- **state_dict()** — base implementation merges `get_sub_blocks()` state with key prefix `"{name}.{key}"`. Override to add own state, then `return _state_dict_with_sub_blocks(self, out)`.
- **load_state_dict()** — base dispatches prefixed keys to the corresponding sub-block; remaining keys are validated (strict) or ignored. So composite blocks get recursive save/load without extra code.
