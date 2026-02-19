# DONE 03: Graph Engine (Executor, Validation, Save/Load) — CLOSED

**Canon:** [WorldGenerator_2.0/TODO_03_GRAPH_ENGINE.md](../WorldGenerator_2.0/TODO_03_GRAPH_ENGINE.md), [Graph_Level.md](../WorldGenerator_2.0/Graph_Level.md).

**Status: TODO_03 is closed.** All required and desirable items are implemented. The graph layer is ready for the next stage (TODO_04 Pipeline).

This document describes what was implemented for the graph engine: executor (run with DAG and cycles), validation with `ValidationResult`, execution plan caching, `Graph.save(save_dir)` / `Graph.load(save_dir)`, `graph.to(device)`, `infer_exposed_ports()`, checkpoint `backend=` API, `from_template(..., validate=)`, and how it prepares for TODO_04 (pipeline), multi-endpoint, and distributed training.

---

## 1. What Was Done

### 1.1 Executor

- **`run(graph, inputs, *, training=False, num_loop_steps=None, device=None, callbacks=None) -> outputs`**  
  Single entrypoint for graph execution (canon §4.2). Implemented in `yggdrasill.executor.run`.
- **Execution order:**  
  - Build directed graph from nodes and edges; compute **strongly connected components (SCC)** (Tarjan).  
  - Topological order of SCCs (sources first).  
  - For each SCC: if singleton and no self-loop → one step `("node", node_id)`; else → step `("cycle", (repr, set of node_ids))`.  
  - Run: execute all `"node"` steps in order; for each `"cycle"` step run the nodes in the SCC (sorted by node_id) **N** times, where N = `num_loop_steps` or `graph.metadata["num_loop_steps"]` (default 1).
- **Buffer:** key `(node_id, port_name)`.  
  - Before execution: fill buffer from `inputs` using `get_input_spec()` (key = exposed name or `"node_id:port_name"`).  
  - For each node: gather inputs from buffer (in-edges **and** pre-filled entries for that node’s input ports, e.g. exposed inputs). Call `block.forward(inputs)`, write outputs to buffer.  
  - After execution: build `outputs` from buffer using `get_output_spec()` (same key convention).
- **training:** if `True`, call `block.train(True)` where supported; else `block.eval()`.
- **device:** if given, call `block.to(device)` for blocks that implement it.
- **callbacks:** list of `hook(node_id, phase, **kwargs)` with `phase` in `"before"`, `"after"`, `"loop_start"`, `"loop_end"` (kwargs include `buffer`, `inputs`, `outputs`, `iteration`, `num_steps` as relevant).
- **dry_run:** if `True`, do not call `block.forward()`; write `None` for output ports so order and buffer keys are validated (canon §9.2).
- **Optional ports:** for optional input ports not in buffer, use default from `block.config.get("defaults", {}).get(port_name)` when present (canon §4.1).
- **Caching:** execution plan is cached by `(id(graph), graph._execution_version)`. `_execution_version` is incremented on `add_node` and `add_edge`, so plan is recomputed only when structure changes.

### 1.2 Graph API Extensions

- **`graph.run(inputs, **kwargs)`** — delegates to `yggdrasill.executor.run(self, inputs, **kwargs)` (same signature).
- **`graph.to(device)`** — calls `block.to(device)` on all blocks that support it; returns `self` for chaining.
- **`graph.save(save_dir)`** — writes `save_dir/config.json` and `save_dir/checkpoint.json` (one-call full save).
- **`Graph.load(save_dir, *, registry=None)`** — class method: load config from `save_dir/config.json` (or `config.yaml` if present), load checkpoint from `save_dir/checkpoint.json`; returns new `Graph` instance.
- **`graph.infer_exposed_ports()`** — set exposed inputs/outputs by inference: input = port with no incoming edge; output = port with no outgoing edge (canon §2.3 optional). Overwrites current exposed I/O.
- **`save_checkpoint(path, format="single"|"dir", backend="json")`** — backend parameter for future "torch"/"safetensors"; only "json" supported for now.
- **`load_from_checkpoint(..., backend="json")`** — same backend contract.
- **`Graph.from_template(template_name, *, registry=None, validate=False, **kwargs)`** — if validate=True, call graph.validate(strict=True) after build.
- **`Graph.from_config(config, registry=None, validate=False)`** — if `validate=True`, call `g.validate(strict=True)` after build (canon §7.3). Node config may be `{"ref": "path/to/file.yaml"}` to load config from file (canon §7.1); `_resolve_config_ref` handles YAML/JSON.
- **`save_checkpoint(path, format="single"|"dir")`** — `"single"` (default): one JSON at `path`; `"dir"`: `path` is a directory, write `path/node_id.json` per node (canon §6.1–6.2). Node IDs must not contain path separators when using `format="dir"`.
- **`load_from_checkpoint(..., checkpoint_dir=None)`** — if `checkpoint_dir` is set, load state from `checkpoint_dir/node_id.json` files and call `load_state_dict` (canon §6.3).
- **`graph.validate(strict=True) -> ValidationResult`** — returns `ValidationResult(errors, warnings)`. If `strict=True` and there are errors, raises `ValueError`; otherwise returns result. Checks: required input ports have an edge or are exposed; optionally (when nodes/exposed exist): reachability from exposed inputs and to exposed outputs (warnings only).
- **`ValidationResult`** — dataclass with `errors: List[str]`, `warnings: List[str]`, and `is_valid` (true iff no errors).

### 1.3 Design for Later TODOs

- **Pipeline (TODO_04):** Pipeline can call `run(graph, inputs)` per graph; input/output contract by name is stable.
- **Multi-endpoint (TODO_07):** Same contract (`get_input_spec` / `get_output_spec`, run → dict by name) allows deploying a graph as a service without changing executor.
- **Distributed / custom backends:** Checkpoint format (e.g. `.pt`, safetensors) can be made pluggable later; executor does not depend on it. `trainable_parameters()` already filters by node; DDP/wrapping can be added at pipeline/stage level.
- **Custom models, training, multiple endpoints:** Core run() and graph contract are kept minimal so extensions (LoRA, adapters, multi-GPU) can wrap blocks or graphs without rewriting the executor.

---

## 2. Code Layout

```
yggdrasill/
  executor/
    __init__.py   # exports run
    run.py        # _scc_tarjan, _topological_order_sccs, _build_execution_plan,
                  # _gather_inputs_for_node (edges + pre-filled buffer),
                  # run(...), execution plan cache
  foundation/
    graph.py      # run(), to(device), save(save_dir), load(save_dir),
                  # save_checkpoint(path, format=), load_from_checkpoint(checkpoint_dir=),
                  # from_config(..., validate=), _resolve_config_ref,
                  # validate(strict) -> ValidationResult, _reachable_from, _reaches_exposed_outputs,
                  # _execution_version, ValidationResult
```

Public API:

- `from yggdrasill.executor import run`  
- `from yggdrasill.foundation import Graph, ValidationResult`  
- `from yggdrasill import run_graph`

---

## 3. Tests

- **tests/foundation/test_graph.py:**  
  - `test_run_dag_graph` — DAG A (add) → B (identity), expose inputs/outputs by name, run and check output.  
  - `test_run_with_callbacks` — run with one callback, assert `before`/`after` invoked.  
  - `test_validate_returns_validation_result` — `validate(strict=False)` returns `ValidationResult` with no errors.  
  - `test_validate_strict_raises_on_errors` — required port missing → raise.  
  - `test_save_and_load_directory` — save graph to dir, load, run and check result (including checkpoint).  
  - `test_graph_to_device_no_raise` — `graph.to("cpu")` / `to(None)` do not raise when blocks have no `.to`.  
  - `test_run_dry_run` — `run(..., dry_run=True)` returns outputs with `None`, no forward calls.  
  - `test_from_config_validate` — `from_config(..., validate=True)` raises when required port has no edge; `validate=False` builds without raising.  
  - `test_save_checkpoint_dir_format` — `save_checkpoint(path, format="dir")` and `load_from_checkpoint(checkpoint_dir=...)` roundtrip.
  - `test_infer_exposed_ports` — infer_exposed_ports() sets exposed I/O from ports with no in/out edges.
  - `test_save_checkpoint_backend_json_only` — backend="json" works; backend="torch" raises NotImplementedError.
  - `test_get_input_output_spec_include_dtype` — get_input_spec(include_dtype=True) / get_output_spec(include_dtype=True) add dtype.

All 67 tests pass.

---

## 4. Usage Example

```python
from yggdrasill.foundation import Graph
from yggdrasill.foundation.registry import BlockRegistry

# Build and run
g = Graph("my_graph")
g.add_node("A", "add", config={"offset": 2}, registry=registry)
g.add_node("B", "identity", registry=registry)
g.add_edge(Edge("A", "out", "B", "x"))
g.expose_input("A", "a", name="a_in")
g.expose_input("A", "b", name="b_in")
g.expose_output("B", "y", name="result")

g.validate()
out = g.run({"a_in": 10, "b_in": 5})
assert out["result"] == 17  # 10 + 5 + 2

# Save / load
g.save("/path/to/save_dir")
g2 = Graph.load("/path/to/save_dir", registry=registry)
out2 = g2.run({"a_in": 1, "b_in": 2})

# Optional: device, callbacks, dry_run
g.to("cpu")
g.run({"a_in": 0, "b_in": 0}, callbacks=[lambda nid, phase, **kw: print(nid, phase)])
g.run({"a_in": 0, "b_in": 0}, dry_run=True)  # validate order and buffer without calling forward

# Checkpoint per-node (canon §6)
g.save_checkpoint("/path/to/ckpt_dir", format="dir")
g2 = Graph.from_config(g.to_config(), registry=registry)
g2.load_from_checkpoint(checkpoint_dir="/path/to/ckpt_dir")
```

---

## 5. Summary

TODO_03 is **closed**: executor (topological order, SCC cycles, buffer, optional port defaults, dry_run), validation with `ValidationResult` and reachability, `save`/`load` directory API, `save_checkpoint(format=, backend=)` and `load_from_checkpoint(..., backend=)`, `from_config(..., validate=)` and config `ref` resolution, `from_template(..., validate=)`, `infer_exposed_ports()`, and `to(device)`. The design keeps the graph contract and single `run()` entrypoint so that pipeline (TODO_04), multi-endpoint, and training extensions can build on this with minimal rewriting. **Ready for next stage.**

---

## 6. Что ещё не сделано на этом этапе

- **Пул повторного использования при сборке графа:** при add_node и from_config несколько узлов с одним и тем же checkpoint_ref (или model_id для API) создают несколько экземпляров блока; движок графа не использует общий пул «ref → блок». См. [MODEL_REUSE.md](../WorldGenerator_2.0/MODEL_REUSE.md).

---

## 7. Что нужно реализовать (по TODO_03 и канону)

- При сборке графа (add_node, from_config): вести пул «checkpoint_ref / model_id → блок»; при создании узла с уже встречавшимся ref подставлять существующий блок вместо нового, чтобы в памяти был один экземпляр модели на один ref. При обучении графа с общими блоками (дедупликация параметров в оптимизаторе, градиенты) — см. [TRAINING_REUSE_AND_API_SCENARIOS.md](../WorldGenerator_2.0/TRAINING_REUSE_AND_API_SCENARIOS.md).

Сводка по этапам: [СТАТУС_ПО_ЭТАПАМ.md](СТАТУС_ПО_ЭТАПАМ.md).
