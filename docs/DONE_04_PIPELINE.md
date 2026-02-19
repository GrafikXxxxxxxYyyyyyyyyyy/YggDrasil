# DONE 04: Pipeline (Graph of Graphs) — CLOSED

**Canon:** [WorldGenerator_2.0/TODO_04_PIPELINE.md](../WorldGenerator_2.0/TODO_04_PIPELINE.md), [Pipeline_Level.md](../WorldGenerator_2.0/Pipeline_Level.md).

**Status: TODO_04 is closed.** Pipeline layer is implemented: structure (Pipeline, PipelineEdge), builder API (add_graph, add_edge, expose_input/expose_output), validation (DAG, reachability), executor (run_pipeline), serialization (to_config/from_config, save/load, checkpoints), trainable nodes, infer_exposed_ports. Ready for Stage (TODO_05) and multi-endpoint.

This document describes what was implemented for the pipeline level and how it fits with the graph contract and future stages.

---

## 1. What Was Done

### 1.1 Data Model

- **Pipeline** — container: (1) **nodes:** `pipeline_node_id: str → Graph`; (2) **edges:** list of `PipelineEdge(source_node, source_port, target_node, target_port)` where ports are **external** graph ports (keys from `get_output_spec` / `get_input_spec`); (3) **exposed inputs/outputs:** list of `{pipeline_node_id, port_name, name?}` for pipeline-level I/O; (4) `pipeline_id`, `_metadata`, `_execution_version`.
- **PipelineEdge** — frozen dataclass; empty strings for any field raise in `__post_init__`.
- **Indices:** `_in_edges_by_node`, `_out_edges_by_node` for fast lookup; `get_edges()`, `get_edges_in(nid)`, `get_edges_out(nid)`.

### 1.2 Builder API

- **add_graph(graph_or_config, pipeline_node_id=None, *, registry=None)**  
  - `graph_or_config`: `Graph` instance, config dict, or path (YAML/JSON). For dict/path, graph is built via `Graph.from_config` / `Graph.from_yaml`.  
  - `pipeline_node_id`: optional; default from `graph.graph_id` or `graph_id` in config or generated `"graph_N"`.  
  - Duplicate `pipeline_node_id` raises `ValueError`.
- **add_edge(source_node_id, source_port, target_node_id, target_port)**  
  - Validates: both nodes exist; `source_port` in `get_output_spec(source_graph)`; `target_port` in `get_input_spec(target_graph)`. Port keys use same convention as graph (name or `node_id:port_name`). Duplicate edge is ignored (no-op).
- **expose_input(pipeline_node_id, port_name, name=None)** / **expose_output(...)**  
  - `port_name` = key from graph’s get_input_spec/get_output_spec. `name` = key for `run(inputs)` / `outputs`; if not set, key is `pipeline_node_id:port_name`.

### 1.3 Contract (Stage/World Compatible)

- **get_input_spec(include_dtype=False)** / **get_output_spec(include_dtype=False)** — same contract as Graph: list of entries with `pipeline_node_id`, `port_name`, optional `name` (and `dtype` when `include_dtype=True`). Keys for run: `name` or `pipeline_node_id:port_name`.
- **run(inputs, *, training=False, device=None, callbacks=None, **graph_run_kwargs) -> outputs** — delegates to `run_pipeline`; single entrypoint for execution (same as Graph for Stage/World).

### 1.4 Validation

- **validate(strict=True, validate_graphs=False) -> ValidationResult**  
  - **DAG:** Kahn topological sort; if not all nodes ordered → error "Pipeline graph has a cycle (not a DAG)".  
  - **Reachability:** from exposed inputs and to exposed outputs (warnings only).  
  - **validate_graphs=True:** call `graph.validate(strict=True)` per node; errors prefixed with node id.  
  - If `strict` and there are errors, raises `ValueError`.

### 1.5 Executor

- **run_pipeline(pipeline, inputs, *, training=False, device=None, callbacks=None, **graph_run_kwargs)** in `yggdrasill.pipeline.run`:  
  - Fill buffer from pipeline `get_input_spec()` and `inputs` (key = name or `pipeline_node_id:port_name`).  
  - Topological order via Kahn.  
  - For each node: gather graph inputs from buffer (in-edges + pre-filled exposed); call `graph.run(graph_inputs, training=..., **graph_run_kwargs)`; write graph outputs to buffer by (pipeline_node_id, port_key).  
  - Build outputs from buffer using `get_output_spec()`.  
- **Callbacks:** `hook(node_id, phase, **kwargs)` with `phase` in `"before"`, `"after"` (same as graph executor).

### 1.6 Serialization and Checkpoints

- **to_config()** / **from_config(config, registry=None)** — graphs stored as nested configs (`graph_config` per node); same schema version and metadata as in canon.
- **state_dict()** / **load_state_dict(state)** — aggregate of graph state dicts by pipeline node id.
- **save_config(path)** / **save_checkpoint(path, format="dir", backend="json")** — checkpoint dir: `path/<node_id>/config.json` and `path/<node_id>/checkpoint.json`; `os.makedirs(sub, exist_ok=True)` before writing each graph checkpoint.
- **load_from_checkpoint(config_path=..., checkpoint_path=..., checkpoint_dir=..., backend="json")** — when `checkpoint_dir` is set, load only weights from `checkpoint_dir/<nid>/checkpoint.json` into existing graphs (no config rebuild).
- **save(save_dir)** / **load(save_dir, registry=None)** — write `save_dir/config.json` and `save_dir/checkpoints/<nid>/`; load config then load checkpoints for each node.

### 1.7 Other

- **infer_exposed_ports()** — set exposed inputs = graph input ports with no incoming pipeline edge; exposed outputs = graph output ports with no outgoing pipeline edge.
- **trainable_parameters()** — yield from trainable graphs only. **set_trainable(pipeline_node_id, bool)**.
- **to(device)** — delegate to each graph.

### 1.8 Single-Graph Pipeline

- Pipeline may have one node and no edges; exposed I/O = that graph’s I/O. Execution: one `graph.run(inputs)`. Same interface as multi-graph pipeline and compatible with Stage (stage node = pipeline of one graph).

---

## 2. Code Layout

```
yggdrasill/
  pipeline/
    __init__.py    # Pipeline, PipelineEdge
    pipeline.py    # Pipeline class, PipelineEdge, add_graph, add_edge, expose_*,
                   # get_input_spec, get_output_spec, validate, infer_exposed_ports,
                   # to_config, from_config, save, load, save_checkpoint, load_from_checkpoint,
                   # trainable_parameters, set_trainable, to(device), run()
    run.py         # _topological_order, run_pipeline, buffer and graph.run delegation
  foundation/
    graph.py       # (unchanged contract: get_input_spec, get_output_spec, run)
```

Public API:

- `from yggdrasill.pipeline import Pipeline, PipelineEdge`
- `from yggdrasill.pipeline.run import run_pipeline`
- `pipeline.run(inputs, ...)` — main execution entrypoint

---

## 3. Tests

- **tests/pipeline/conftest.py** — `registry` fixture (add, identity).
- **tests/pipeline/test_pipeline.py:**  
  - add_graph: instance, auto node_id, from config, duplicate raises  
  - add_edge: valid edge, unknown source/target port raises, unknown node raises  
  - validate: DAG ok, cycle error, strict raises, reachability warnings  
  - run: single graph, two graphs (G1→G2), callbacks  
  - get_input_spec / get_output_spec  
  - to_config / from_config roundtrip  
  - save / load directory (with checkpoint)  
  - trainable_parameters, set_trainable  
  - infer_exposed_ports  
  - PipelineEdge empty string raises  

All 88 tests pass (67 foundation/task_nodes + 21 pipeline).

---

## 4. Relation to Stage and Multi-Endpoint

- **Stage (TODO_05):** Stage = graph of pipelines (or sub-stages). Same contract: get_input_spec, get_output_spec, run(inputs). Pipeline is the executable unit; Stage will call pipeline.run() per node.
- **Multi-endpoint (canon §9):** Pipeline and graph share the same run contract; endpoint_url per node can be added later for remote execution without changing this API.
- **Dry-run for pipeline:** Not implemented in this TODO; can be added (run without calling graph.run, only validate order and buffer keys).

---

## 5. Summary

TODO_04 is **closed**: Pipeline as graph of graphs, PipelineEdge, add_graph (Graph/config/path), add_edge with port validation, expose_input/expose_output, get_input_spec/get_output_spec, validate (DAG, reachability, optional graph.validate), run_pipeline (topological order, buffer, graph.run), to_config/from_config, save/load and checkpoints (dir per node), trainable_parameters/set_trainable, infer_exposed_ports. Single-graph pipeline is supported. **Ready for TODO_05 (Stage).**

---

## 6. Что ещё не сделано на этом этапе

- **Повторное использование модели в пайплайне:** если два графа-узла пайплайна используют один и тот же checkpoint_ref (или одну и ту же модель по model_id), в памяти создаются два набора весов; пула «один ref → один экземпляр» на уровне пайплайна (или при загрузке из конфига/чекпоинта) нет. См. [MODEL_REUSE.md](../WorldGenerator_2.0/MODEL_REUSE.md).

---

## 7. Что нужно реализовать (по TODO_04 и канону)

- При add_graph и from_config/load_from_checkpoint: использовать общий пул checkpoint_ref (и model_id для API); два узла пайплайна с одним ref не должны дублировать экземпляр модели — один ref → один экземпляр (граф/блоки) в памяти. При обучении пайплайна с общими графами (коллизии, LoRA на общий блок) — см. [TRAINING_REUSE_AND_API_SCENARIOS.md](../WorldGenerator_2.0/TRAINING_REUSE_AND_API_SCENARIOS.md).

Сводка по этапам: [СТАТУС_ПО_ЭТАПАМ.md](СТАТУС_ПО_ЭТАПАМ.md).
