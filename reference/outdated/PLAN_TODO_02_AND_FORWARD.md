# Plan: TODO_02 Completion and Forward Stages (WorldGenerator_2.0)

This document summarizes the **WorldGenerator_2.0** canon, what is **done** for TODO_02, what **remains** for TODO_02, and what **later stages (TODO_03–07)** require from the graph/task-nodes level so we don’t block them.

**Project canon:** [CANON.md](../WorldGenerator_2.0/CANON.md) — all-encompassing architecture reference (WorldGenerator_2.0/).

---

## 1. Canon Overview (WorldGenerator_2.0)

| Doc | Purpose |
|-----|--------|
| **TODO_01** | Foundation: Block, Port, Node, Edge, Graph (structure), Registry. |
| **TODO_02** | Task nodes (abstract types + role rules) and graph level: AddNode(block_or_type, auto_connect), AddEdge, expose I/O, serialization, from_config/from_template, validation, trainable_parameters. Executor (run) is described in TODO_02 B.3 but detailed in **TODO_03**. |
| **TODO_03** | Graph engine: executor (topology, SCC/cycles, buffer, run(graph, inputs)), full validation, save_config/save_checkpoint to disk, from_yaml. |
| **TODO_04** | Pipeline: nodes = graphs; edges between graph I/O; run(pipeline, inputs). |
| **TODO_05** | Stage: nodes = pipelines; state contract; run(stage, state). |
| **TODO_06** | World: nodes = stages; cycle (Philosophier → Author → …); run(world, state, action?). |
| **TODO_07** | Future: modalities, backends, observability, config schemas. |

Supporting canon: **Abstract_Task_Nodes.md**, **Graph_Level.md**, **SERIALIZATION_AT_ALL_LEVELS.md**, **TRAINABILITY_AT_ALL_LEVELS.md**, **MULTI_ENDPOINT_DEPLOYMENT.md**, **Pipeline_Level.md**, **Stage_Level.md**, **World_Level.md**.

---

## 2. TODO_02 — What Is Done

- **Part A — Abstract task nodes**
  - Roles: `ROLE_*`, `role_from_block_type()`, `KNOWN_ROLES`.
  - Role rules: table (source_role, target_role) → [(src_port, dst_port)]; `get_rule_edges()`, `suggest_edges_for_new_node()`.
  - Abstract classes: `AbstractBackbone`, `AbstractSolver`, `AbstractCodec`, `AbstractConditioner`, `AbstractTokenizer`, `AbstractAdapter`, `AbstractGuidance` (ports + abstract `forward()`).
  - Stub blocks: identity blocks per role, registered as `role/identity`; `register_task_node_stubs()`.
- **Part B — Graph level**
  - `add_node(block_or_type, node_id=..., config=..., registry=..., auto_connect=...)` (Node, block instance, or block_type string); returns node_id; `auto_connect_fn` callback (task_nodes sets it via `use_task_node_auto_connect`).
  - `add_edge(Edge)` with port/node validation (foundation).
  - Exposed inputs/outputs, `get_input_spec()` / `get_output_spec()` (foundation).
  - `to_config()` / `from_config()` (structure only); `state_dict()` / `load_state_dict()`; `load_from_checkpoint(config_path=..., checkpoint_path=..., config=..., checkpoint=..., registry=...)` (in-memory or from JSON files).
  - **Обучаемость:** `trainable_parameters()` — только узлы с `trainable=True`; в конфиге у каждого узла поле `trainable`; API `set_trainable(node_id, bool)` (TRAINABILITY_AT_ALL_LEVELS, TODO_02 B.5).
  - `validate()`: обязательные входы имеют входящее ребро или exposed.
  - `from_template(template_name, **kwargs)`; task_nodes регистрирует `"text_to_image"`.
  - **Сохранение на диск:** `save_config(path)`, `save_checkpoint(path)`, `from_yaml(path)`.
  - **Расширяемость:** в конфиге опционально `graph_kind` и `metadata`; роли дополнены `position_embedder`, `llm`, `vlm`. Расписание (noise schedule) — зона ответственности Solver, отдельной роли NoiseSchedule нет.

---

## 3. TODO_02 — Что остаётся (опционально / отложено)

- **Политика неоднозначности auto_connect:** при нескольких кандидатах (несколько conditioner’ов) — документировать поведение (например, первый подходящий); при необходимости конфигурируемо позже.
- **Исполнитель (run):** реализуется в **TODO_03** (графовый движок), не в TODO_02.
- **Абстрактные классы PositionEmbedder (A.9):** роль добавлена в KNOWN_ROLES; при появлении сценария — абстракт и стаб. NoiseSchedule не вводим: расписание целиком входит в роль Solver.

---

## 4. What TODO_03–07 Need from Graph / Task Nodes

- **TODO_03 (Graph engine)**  
  - From TODO_02: Graph with nodes, edges, exposed I/O, `to_config`/`from_config`, `state_dict`/`load_state_dict`, `trainable_parameters()`, `validate()`, `from_template`, **save_config(path)**, **save_checkpoint(path)**, **from_yaml(path)**.  
  - TODO_03 adds: Executor (topological order, SCC/cycles, buffer, `run(graph, inputs, training=..., num_loop_steps=...)`), full validation (reachability, SCC), optional from_yaml, file format choices for checkpoint.

- **TODO_04 (Pipeline)**  
  - Nodes = **Graph** instances; edges between **graph** exposed ports; `run(pipeline, inputs)` calls `run(graph, inputs)` per node.  
  - Needs: Stable Graph API (`get_input_spec`, `get_output_spec`, `run`, `from_config`, `save_config`, `save_checkpoint`, `load_from_checkpoint`).

- **TODO_05 (Stage)**  
  - Nodes = pipelines; state contract; run(stage, state).  
  - Depends on Pipeline (TODO_04).

- **TODO_06 (World)**  
  - Nodes = stages; cycle; run(world, state, action?).  
  - Depends on Stage (TODO_05).

- **TODO_07**  
  - Modalities, backends, observability, config schemas; no new graph-level contracts.

---

## 5. Implemented in This Pass (TODO_02 Completion)

- **Graph.save_config(path: str)**  
  - Writes `to_config()` to a JSON file (UTF-8). Ensures structure is serializable to disk for TODO_03 and pipeline/stage/world configs.

- **Graph.save_checkpoint(path: str)**  
  - Writes `state_dict()` to a JSON file. Weights are JSON-serializable for current blocks (e.g. AddBlock offset); for PyTorch tensors, TODO_03 can add a different format (e.g. .pt / safetensors) and still use the same API.

- **Graph.from_yaml(path: str, registry=None)**  
  - Class method: load config from file (YAML or JSON via OmegaConf), then `from_config(..., registry=registry)`. Matches canon “from_config or from_yaml” and supports pipeline/stage/world config loading.

These complete the “save/load graph to disk” contract expected by SERIALIZATION_AT_ALL_LEVELS and by TODO_03/04 (pipeline saves graph configs and checkpoints).

---

## 6. Расширяемость: не-диффузия, не-нейросети, VLM, LLM (SCALABILITY_AND_EXTENSIBILITY)

- **Граф не привязан к типу модели:** контракт — порты и `forward`; блок может быть диффузией, LLM, VLM, классификатором, чисто данныхым пайплайном (без весов). `state_dict()` может быть пустым (не-нейросеть).
- **graph_kind:** опциональная подсказка для executor/пайплайна (`diffusion`, `llm`, `vlm`, `data`, `generic`). Исполнитель может использовать для `num_loop_steps`, стриминга и т.д.
- **metadata:** произвольный словарь (версия, описание, параметры run); сериализуется в конфиг.
- **Роли:** помимо backbone, solver, codec, … добавлены `position_embedder`, `llm`, `vlm`. Расписание (sigma/alpha от timestep) — ответственность Solver, отдельной роли «NoiseSchedule» нет. Расширения регистрируют блоки (напр. `llm/llama`, `vlm/llava`); при добавлении правил в role_rules автосвязывание работает без изменения ядра.
- **Масштабируемость:** новый тип блока = класс + регистрация; граф поддерживает любые архитектуры (LLM, VLM, диффузия, не-нейросеть) через единый контракт блоков.

---

## 7. File / Doc References

- **Canon:** [CANON.md](../WorldGenerator_2.0/CANON.md); [TODO_01](../WorldGenerator_2.0/TODO_01_FOUNDATION.md)…[TODO_07](../WorldGenerator_2.0/TODO_07_FUTURE_AND_IMPROVEMENTS.md), [Abstract_Task_Nodes.md](../WorldGenerator_2.0/Abstract_Task_Nodes.md), [Graph_Level.md](../WorldGenerator_2.0/Graph_Level.md), [AGENT_SYSTEMS_SUPPORT.md](../WorldGenerator_2.0/AGENT_SYSTEMS_SUPPORT.md), [EXPANSION_UNIVERSE_GRAPH_OF_WORLDS.md](../WorldGenerator_2.0/EXPANSION_UNIVERSE_GRAPH_OF_WORLDS.md), [Scheme.md](../WorldGenerator_2.0/Scheme.md).
- **Code:** `yggdrasill/foundation/`, `yggdrasill/task_nodes/`.
- **Docs:** [DONE_01_FOUNDATION.md](DONE_01_FOUNDATION.md), [DONE_02_TASK_NODES_AND_GRAPH.md](DONE_02_TASK_NODES_AND_GRAPH.md), [OUTLINE_TODO_02_TO_07.md](OUTLINE_TODO_02_TO_07.md).
- **Канон расширяемости:** [SCALABILITY_AND_EXTENSIBILITY.md](../WorldGenerator_2.0/SCALABILITY_AND_EXTENSIBILITY.md), [TRAINABILITY_AT_ALL_LEVELS.md](../WorldGenerator_2.0/TRAINABILITY_AT_ALL_LEVELS.md), [TODO_07_FUTURE_AND_IMPROVEMENTS.md](../WorldGenerator_2.0/TODO_07_FUTURE_AND_IMPROVEMENTS.md).
