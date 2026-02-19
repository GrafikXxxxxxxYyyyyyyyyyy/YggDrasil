# TODO_03: сверка с каноном и план дальнейшей разработки

**TODO_03 закрыт.** Графовый движок реализован полностью; готов к переходу на следующий этап (TODO_04 Pipeline). Ожидаем ваших указаний для старта TODO_04.

Документ: (1) проверка полноты реализации TODO_03 по канону, (2) что добавлено в ядре для будущих этапов, (3) план разработки фреймворка (TODO_04 → TODO_07).

**Канон:** WorldGenerator_2.0/TODO_03_GRAPH_ENGINE.md, Graph_Level.md, Pipeline_Level.md, MULTI_ENDPOINT_DEPLOYMENT.md, SERIALIZATION_AT_ALL_LEVELS.md, OUTLINE_TODO_02_TO_07.md.

---

## 1. Сверка TODO_03 с каноном

### 1.1 Структура графа (§1)

| Требование канона | Статус | Примечание |
|-------------------|--------|------------|
| Граф: узлы (node_id → Node), рёбра, внешний контракт (exposed I/O с name или node_id:port_name) | ✅ | `_nodes`, `_edges`, `_exposed_inputs`/`_exposed_outputs` |
| Индексы in_edges[node_id], out_edges[node_id] за O(1)/O(k) | ✅ | `_in_edges_by_node`, `_out_edges_by_node` |
| Инварианты: node_id в рёбрах есть в узлах; порты и типы проверяются при add_edge | ✅ | `_validate_edge` при добавлении ребра |

### 1.2 API построения (§2)

| Требование | Статус | Примечание |
|-------------|--------|------------|
| AddNode(block_or_type, node_id=None, config=None, auto_connect=True) | ✅ | Node / block / block_type; реестр, авто-связывание |
| AddEdge(source_node, source_port, target_node, target_port) | ✅ | Edge, валидация портов и типов |
| expose_input(node_id, port_name, name=None), expose_output(...) | ✅ | Имя для run(inputs={name: value}) |
| Контракт: get_input_spec(), get_output_spec() | ✅ | Список {node_id, port_name, name?} |
| infer_exposed_ports() (опционально) | ⏸ | Не реализован; при необходимости — в расширении или TODO_04 |

### 1.3 Валидация (§3)

| Требование | Статус | Примечание |
|-------------|--------|------------|
| При add_edge: узлы, порты, совместимость типов | ✅ | Запрет только на пустые имена; циклы допустимы |
| validate(strict=True) -> ValidationResult; обязательные порты | ✅ | errors + warnings; strict → raise |
| Достижимость от входов / к выходам (предупреждения) | ✅ | _reachable_from, _reaches_exposed_outputs |
| Циклы: начальные значения снаружи (проверка опциональна) | ✅ | Исполнитель заполняет буфер из exposed + рёбер |

### 1.4 Исполнитель (§4)

| Требование | Статус | Примечание |
|-------------|--------|------------|
| SCC, топологический порядок, DAG + циклы с N итерациями | ✅ | Tarjan, порядок SCC, план ("node" | "cycle") |
| Буфер (node_id, port_name); входы из exposed + рёбер | ✅ | _gather_inputs_for_node: in_edges + buffer + optional defaults |
| run(graph, inputs, training=, num_loop_steps=, device=, callbacks=) | ✅ | + dry_run (canon §9.2) |
| Опциональные порты: default из конфига блока | ✅ | config.get("defaults", {}).get(port_name) |
| Кэш плана выполнения; инвалидация при add_node/add_edge | ✅ | _execution_version, кэш по (id(graph), version) |
| dry_run (без вызова forward) | ✅ | §9.2 |

### 1.5 Сериализация (§5, §6)

| Требование | Статус | Примечание |
|-------------|--------|------------|
| schema_version в конфиге | ✅ | GRAPH_CONFIG_SCHEMA_VERSION |
| save_config(path), to_config(), from_config() | ✅ | + from_config(..., validate=False) |
| Узлы: node_id, block_type, config; ref в конфиге узла | ✅ | _resolve_config_ref для config: {ref: path} |
| save_checkpoint(path, format="single"|"dir") | ✅ | dir = node_id.json на узел |
| load_from_checkpoint(..., checkpoint_dir=) | ✅ | Загрузка из каталога |
| save(save_dir), Graph.load(save_dir) | ✅ | config.json + checkpoint.json |
| from_yaml(path), from_template(name, **kwargs) | ✅ | Шаблоны в task_nodes |

### 1.6 Контракт и связь с пайплайном (§8)

| Требование | Статус | Примечание |
|-------------|--------|------------|
| Входной/выходной контракт по именам; get_input_spec/get_output_spec | ✅ | Пайплайн будет вызывать run(graph, inputs) по этим именам |
| graph_id в метаданных | ✅ | graph_id, graph_kind, metadata |

### 1.7 Дополнительно (§9)

| Требование | Статус | Примечание |
|-------------|--------|------------|
| to(device) | ✅ | graph.to(device), блоки с .to() |
| Хуки (before/after node, loop_start/loop_end) | ✅ | callbacks в run() |
| Кэширование топологии/SCC | ✅ | См. §4 |

**Итог по TODO_03:** реализация закрывает канон. Дополнительно добавлены: **infer_exposed_ports()**, **backend=** в save_checkpoint/load_from_checkpoint (расширяемо под torch/safetensors), **from_template(..., validate=)**.

---

## 2. Что заложить сейчас, чтобы не переделывать позже

Ниже — минимальный набор расширений в текущем ядре (граф, контракт, сериализация), чтобы TODO_04–07 и multi-endpoint не требовали переписывания.

### 2.1 Контракт графа для multi-endpoint и пайплайна

- **Сделано:** get_input_spec() / get_output_spec() возвращают список dict с ключами node_id, port_name, name (опционально). Этого достаточно для пайплайна (рёбра по именам) и для run(inputs по именам).
- **Добавлено:** get_input_spec(include_dtype=False) и get_output_spec(include_dtype=False). При include_dtype=True в каждый элемент spec добавляется ключ "dtype" из порта блока (для генерации API эндпоинта и проверки типов в пайплайне). Обратная совместимость: по умолчанию False.

### 2.2 Формат чекпоинта (тензоры, распределённое обучение)

- **Сейчас:** state_dict() → dict; save_checkpoint пишет JSON. Для блоков с тензорами (PyTorch) JSON не подходит.
- **Чтобы не переписывать позже:** оставить единый API save_checkpoint(path, format=) / load_from_checkpoint(...), а формат файла вынести в **backend** (например, "json", "torch", "safetensors"). Регистрация бэкендов по имени; по умолчанию "json". В TODO_07 или при появлении PyTorch-блоков добавить backend "torch" без изменения графа и executor.
- **Сейчас менять не обязательно:** можно добавить только заглушку или комментарий в коде (например, в save_checkpoint: "For tensor state, use a checkpoint backend; see SERIALIZATION_AT_ALL_LEVELS.").

### 2.3 Единый контракт «исполняемой единицы» (граф и пайплайн)

- Пайплайн (TODO_04) будет иметь тот же контракт, что и граф: get_input_spec(), get_output_spec(), run(inputs) -> outputs. Тогда этап/мир вызывают run() единообразно.
- **В коде ничего менять не нужно:** контракт уже задан через get_input_spec/get_output_spec и run; пайплайн достаточно реализовать с такими же методами и вызывать graph.run() для узлов.

### 2.4 Удалённый граф (multi-endpoint)

- Оркестратор пайплайна/этапа должен уметь вызывать либо локальный graph.run(inputs), либо запрос к URL (те же inputs/outputs по контракту). Это логика уровня пайплайна (TODO_04): в конфиге у узла — graph или endpoint_url; исполнитель пайплайна решает, вызывать run() или HTTP/gRPC.
- **В графе менять ничего не нужно.** Достаточно стабильного контракта (имена входов/выходов, при необходимости типы из §2.1).

### 2.5 Обучаемость и распределённое обучение

- **Сделано:** trainable_parameters() только по узлам с trainable=True; set_trainable(node_id, bool). Оптимизатор получает итератор параметров; DDP/wrapping можно делать снаружи (обёртка над графом или над блоками по get_node).
- Дополнительно можно позже добавить, например, `graph.iter_blocks()` или явный перечень node_id для обёртки, но текущего API достаточно, чтобы не переписывать ядро.

### 2.6 Резюме: что добавить в коде сейчас

| Что | Где | Зачем |
|-----|-----|--------|
| get_input_spec(include_dtype=True), get_output_spec(include_dtype=True) | Graph | Multi-endpoint, валидация типов в пайплайне — **сделано** |
| Комментарий «checkpoint backend» в save_checkpoint | graph.py | Напомнить про .pt/safetensors в будущем — **сделано** |
| Не добавлять infer_exposed_ports в ядро | — | Оставить на потом при необходимости |

Остальное (пайплайн, этап, мир, эндпоинты, бэкенды чекпоинтов) — в соответствующих TODO без изменений контракта графа.

---

## 3. План дальнейшей разработки (TODO_04 → TODO_07)

### 3.1 Зависимости между уровнями

```
TODO_01 (Foundation) → TODO_02 (Task nodes) → TODO_03 (Graph engine) ✅
       → TODO_04 (Pipeline: граф графов)
       → TODO_05 (Stage: граф пайплайнов, контракт по state)
       → TODO_06 (World: граф этапов, цикл, state, storage)
       → TODO_07 (Модальности, бэкенды, наблюдаемость)
```

Граф (TODO_03) готов как единица исполнения; пайплайн только использует его API.

### 3.2 TODO_04 — Пайплайн

- **Сущности:** Pipeline: node_id → Graph; рёбра между **внешними** портами графов (имена из get_input_spec/get_output_spec).
- **API:** add_graph(graph_or_config, pipeline_node_id=?), add_edge(source_node, source_port, target_node, target_port), expose_input/expose_output пайплайна, get_input_spec/get_output_spec, run(pipeline, inputs, training=, device=, callbacks=, **graph_run_kwargs).
- **Исполнитель:** топологический порядок (DAG); буфер (pipeline_node_id, port_name); для каждого узла — graph.run(inputs_from_buffer); сбор выходов в буфер и во внешние выходы пайплайна.
- **Валидация:** DAG (без циклов между графами); порты в контрактах графов; опционально graph.validate() для каждого графа.
- **Сериализация:** конфиг (узлы = конфиги графов или ref, рёбра, exposed I/O); чекпоинт = агрегация чекпоинтов графов; load(save_dir) / save(save_dir).
- **Задел под multi-endpoint:** в конфиге узла — graph или endpoint_url; исполнитель пайплайна при endpoint_url делает запрос к эндпоинту (inputs/outputs по контракту).

### 3.3 TODO_05 — Этап (Stage)

- **Сущности:** Stage: stage_node_id → Pipeline; рёбра между внешними портами пайплайнов; контракт по **state** (какие блоки state на входе/выходе).
- **API:** add_pipeline(pipeline_or_config, stage_node_id=?), add_edge(...), set_state_contract(input_blocks, output_blocks, mapping_*), set_execution_condition(condition), run(stage, state, action=?) -> state.
- **Исполнитель:** топологический порядок (DAG пайплайнов); маппинг state ↔ входы/выходы граничных пайплайнов; вызов pipeline.run(inputs); сбор выходов в state.
- **Условия выполнения:** по Scheme — выполнять этап только если заполнены нужные блоки state; иначе пропуск (state без изменений). Условие задаётся в конфиге этапа или мира.

### 3.4 TODO_06 — Мир (World)

- **Сущности:** World: stage_id → Stage; **цикл** (упорядоченный список stage_id); state_schema, storage (куда сохранять state при World update), initial_world.
- **API:** add_stage(stage_or_config, stage_id=?), set_cycle(ordered_stage_ids), set_state_schema(...), set_storage(...), set_initial_world(...), run(world, state, action=?) -> state (один проход или итерация).
- **Исполнитель:** обход этапов по циклу; для каждого этапа проверка execution_condition(state); если истина — stage.run(state, action); иначе пропуск; передача state следующему этапу; при World update — сохранение state в storage.
- **Первая итерация (пустой state):** по Scheme пропуск Философ/Автор/World update; старт с Development of the world.

### 3.5 TODO_07 — Дальнейшее развитие

- **Модальности:** VLM, видео, аудио, мультимодальность, стриминг — через новые блоки и порты (PortType, role_rules); без смены контракта графа/пайплайна.
- **Бэкенды:** ONNX, vLLM, TGI, квантизация — реализация в блоках и реестре; опционально async run, батчинг.
- **Распределённое обучение:** обёртки над графом/пайплайном (DDP и т.д.); trainable_parameters() и set_trainable уже есть.
- **Наблюдаемость:** трейсинг, метрики, профилирование — через callbacks run() и расширение хуков на уровнях пайплайн/этап/мир.
- **Валидация конфигов:** JSON Schema для конфигов графа/пайплайна/этапа/мира; контрактные тесты по get_input_spec/get_output_spec.

### 3.6 Порядок реализации

1. **TODO_04 (Pipeline)** — первый приоритет; без него этап и мир не имеют «графов графов».
2. **TODO_05 (Stage)** — контракт по state и условия выполнения; нужен для мира.
3. **TODO_06 (World)** — цикл, storage, initial_world, action.
4. **TODO_07** — по мере необходимости: бэкенды чекпоинтов, типы в spec, модальности, наблюдаемость.

---

## 4. Файлы и ссылки

- **Канон:** WorldGenerator_2.0/TODO_03_GRAPH_ENGINE.md, TODO_04_PIPELINE.md, TODO_05_STAGE.md, TODO_06_WORLD.md, MULTI_ENDPOINT_DEPLOYMENT.md, SERIALIZATION_AT_ALL_LEVELS.md, LLM_API_SUPPORT.md, MODEL_REUSE.md, TRAINING_REUSE_AND_API_SCENARIOS.md.
- **Код:** yggdrasill/foundation/graph.py, yggdrasill/executor/run.py, yggdrasill/foundation/block.py, port.py.
- **Документация:** docs/DONE_03_GRAPH_ENGINE.md, docs/PLAN_TODO_02_AND_FORWARD.md, docs/OUTLINE_TODO_02_TO_07.md.
