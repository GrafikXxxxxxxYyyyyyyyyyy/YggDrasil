# DONE 02: Task Nodes and Graph Level

**Canon:** [WorldGenerator_2.0/TODO_02_TASK_NODES_AND_GRAPH.md](../WorldGenerator_2.0/TODO_02_TASK_NODES_AND_GRAPH.md), [Abstract_Task_Nodes.md](../WorldGenerator_2.0/Abstract_Task_Nodes.md), [Graph_Level.md](../WorldGenerator_2.0/Graph_Level.md).

This document describes what was implemented for task nodes and graph-level behaviour: abstract task node types, role rules, auto-connect, add_node(block_or_type), from_template, trainable_parameters, validation, and how it connects to TODO_03 (executor).

---

## 1. What Was Done

### Part A: Abstract Task Nodes

- **Roles** — constants and helpers: `ROLE_BACKBONE`, `ROLE_SOLVER`, `ROLE_CODEC`, `ROLE_CONDITIONER`, `ROLE_TOKENIZER`, `ROLE_ADAPTER`, `ROLE_GUIDANCE`, `KNOWN_ROLES`; `role_from_block_type(block_type)` (e.g. `"backbone/unet2d"` → `"backbone"`).
- **Role rules** — table of default port links `(source_role, target_role) -> [(src_port, dst_port), ...]`: tokenizer→conditioner, conditioner→backbone, backbone→solver, solver→backbone (cycle), codec↔solver, conditioner→adapter, adapter→backbone, backbone→guidance, guidance→solver. `get_rule_edges(src_role, tgt_role)` and `suggest_edges_for_new_node(new_node_id, new_role, existing_roles_by_node_id)` return suggested edges for auto-connect.
- **Abstract task node classes** — each inherits `AbstractBaseBlock` and declares canonical ports and abstract `forward()`: `AbstractBackbone`, `AbstractSolver`, `AbstractCodec`, `AbstractConditioner`, `AbstractTokenizer`, `AbstractAdapter`, `AbstractGuidance`. Codec uses distinct ports: `encode_image`, `encode_latent`, `decode_latent`, `decode_image`.

### Part B: Graph-Level Behaviour

- **Graph.add_node(block_or_type, ...)** — accepts:
  - a `Node` (unchanged behaviour),
  - a block instance (wrapped in Node with generated or given `node_id`),
  - a `block_type` string (block built via `registry.build(...)`).
  Keyword-only: `node_id`, `config`, `registry`, `auto_connect`. Returns `node_id`. When `auto_connect=True`, calls `graph.auto_connect_fn(self, node_id, block)` if set (foundation does not depend on task_nodes).
- **Auto-connect** — in `task_nodes`: `apply_auto_connect(graph, new_node_id, new_block)` uses role rules to suggest edges and adds only those whose ports exist and are compatible; `use_task_node_auto_connect(graph)` sets `graph.auto_connect_fn = apply_auto_connect`.
- **Concrete stub blocks** — identity/passthrough blocks per role: `backbone/identity`, `solver/identity`, `codec/identity`, `conditioner/identity`, `tokenizer/identity`, `adapter/identity`, `guidance/identity`; registered via `@register_block` and `register_task_node_stubs(registry)`.
- **Graph.trainable_parameters()** — возвращает параметры только узлов с `trainable=True`. **set_trainable(node_id, bool)** и поле **trainable** в конфиге узла (TRAINABILITY_AT_ALL_LEVELS, TODO_02 B.5).
- **Graph.from_template(template_name, **kwargs)** — builds graph from a named template; templates are registered by extensions (e.g. `task_nodes`). Template `"text_to_image"` returns a config with tokenizer, conditioner, backbone, solver, codec (identity stubs) and role-rule edges; exposed inputs (text, timestep) and output (image).
- **Graph.validate()** — checks that every required input port has an incoming edge or is exposed as graph input; raises otherwise.
- **from_config** — unchanged; builds graph from config (nodes, edges, exposed_inputs, exposed_outputs).
- **save_config(path)** — writes `to_config()` to a JSON file (structure only).
- **save_checkpoint(path)** — writes `state_dict()` to a JSON file (weights; for tensor state TODO_03 may add other formats).
- **from_yaml(path, registry=None)** — класс-метод: загрузка конфига из YAML/JSON, затем `from_config`.
- **graph_kind** (опционально): подсказка для executor (`diffusion`, `llm`, `vlm`, `data`, `generic`). **metadata**: произвольный словарь в конфиге.
- **Роли:** добавлены константы для расширений — `position_embedder`, `llm`, `vlm`. Расписание (noise schedule) выполняет Solver; отдельной роли NoiseSchedule нет.

---

## 2. How It Is Implemented in Code

### 2.1 Directory and module layout

```
yggdrasill/
  foundation/
    graph.py            # add_node(block_or_type, ...), from_template, trainable_parameters, validate
  task_nodes/
    __init__.py         # re-exports roles, role_rules, abstract, auto_connect; imports stubs + templates
    roles.py            # ROLE_*, role_from_block_type, KNOWN_ROLES
    role_rules.py       # get_rule_edges, suggest_edges_for_new_node
    abstract.py         # AbstractBackbone, AbstractSolver, AbstractCodec, ...
    auto_connect.py     # apply_auto_connect, use_task_node_auto_connect
    stubs.py            # IdentityBackbone, IdentitySolver, ... + register_task_node_stubs
    templates.py        # _text_to_image_config, _register_templates (Graph._template_builders)

tests/
  foundation/
    test_graph.py       # add_node by type/instance, trainable_parameters, validate, from_template
  task_nodes/
    test_roles.py
    test_role_rules.py
    test_auto_connect.py
```

### 2.2 Usage examples

**Сборка задачи в пару строчек (рекомендуемый API):**

```python
import yggdrasill.task_nodes  # регистрирует стабы и шаблоны
from yggdrasill.foundation.graph import Graph

g = Graph()
g.add_node("MyBackbone", "backbone/identity")
g.add_node("MySolver", "solver/identity", pretrained="path/to.ckpt")  # опционально
g.add_node("MyConditioner", "conditioner/identity")
g.add_node("MyControlnet", "adapter/identity")  # адаптер (ControlNet, LoRA и т.д.)
# Рёбра по ролям создаются автоматически (auto_connect=True по умолчанию)
```

Раньше (без скрытия деталей): вызов `use_task_node_auto_connect(g)`, передача `node_id=`, `registry=`, `auto_connect=True` — по-прежнему поддерживаются через один аргумент (block_type) или явные kwargs.

**Build from template:**

```python
import yggdrasill.task_nodes
from yggdrasill.foundation.graph import Graph
from yggdrasill.foundation.registry import BlockRegistry

g = Graph.from_template("text_to_image", registry=BlockRegistry.global_registry())
g.validate()
# g has tokenizer, conditioner, backbone, solver, codec and role-rule edges
```

**Trainable parameters and validation:**

```python
params = list(g.trainable_parameters())  # union of all blocks
g.validate()  # raises if required input port has no edge and is not exposed
```

---

## 3. Connection to TODO_03 (Executor)

- The graph exposes **input/output spec** via `get_input_spec()` and `get_output_spec()`; the executor will feed values to exposed inputs and read from exposed outputs.
- **trainable_parameters()** is intended for the executor (or training loop) to pass to an optimizer.
- **from_config / to_config** allow saving and restoring graph structure; the executor will run the graph by traversing nodes and edges and calling `block.forward(inputs)`.
- **from_template** produces a runnable graph structure; the executor will need to handle multi-step loops (backbone ↔ solver) and initial conditions (e.g. timestep from schedule).

---

## 4. Tests

- **Roles:** `role_from_block_type` for prefix, bare, unknown.
- **Role rules:** `get_rule_edges` for backbone↔solver, conditioner→backbone, codec↔solver; `suggest_edges_for_new_node` returns correct (src_nid, src_port, tgt_nid, tgt_port) list.
- **Auto-connect:** graph with `use_task_node_auto_connect`; add backbone and solver; assert edges between them by role rules.
- **Graph:** `add_node` by block_type, by block instance, with explicit node_id; reject extra kwargs when passing Node; `trainable_parameters` empty for identity blocks; `validate` raises when required port has no edge, passes when exposed or has edge; `from_template("text_to_image")` builds graph and passes validate.

**Plan and forward stages:** See [PLAN_TODO_02_AND_FORWARD.md](PLAN_TODO_02_AND_FORWARD.md) for canon overview, TODO_02 gaps, and what TODO_03–07 require from the graph level.

All tests (foundation + task_nodes) pass, including save_config, save_checkpoint, and from_yaml.

---

## 5. Что ещё не сделано на этом этапе

- **Узлы LLM через API:** типы вида `llm/api`, `llm/openai`, `llm/anthropic` и контракт «промпт/контекст → текст» без локальных весов не реализованы; регистрация и конфиг (provider, model_id) по [LLM_API_SUPPORT.md](../WorldGenerator_2.0/LLM_API_SUPPORT.md) отсутствуют.
- **Повторное использование модели при сборке графа:** при add_node или from_config один и тот же checkpoint_ref (или model_id для API) может приводить к двум разным экземплярам блока; пул «один ref → один экземпляр» при создании узла не используется. См. [MODEL_REUSE.md](../WorldGenerator_2.0/MODEL_REUSE.md).

---

## 6. Что нужно реализовать (по TODO_02 и канону)

- Регистрация типов узлов для LLM (и при необходимости других типов) только через API; конфиг узла с backend: "api", provider, model_id; единый контракт портов для локального и API.
- При add_node / from_config: не создавать второй блок для того же checkpoint_ref или model_id — использовать общий пул (один ref → один экземпляр блока). При обучении графа с повторным использованием и при API-узлах — см. [TRAINING_REUSE_AND_API_SCENARIOS.md](../WorldGenerator_2.0/TRAINING_REUSE_AND_API_SCENARIOS.md).

Сводка по этапам: [СТАТУС_ПО_ЭТАПАМ.md](СТАТУС_ПО_ЭТАПАМ.md).
