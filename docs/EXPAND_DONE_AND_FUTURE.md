# Расширение DONE — Что есть, чего нет, как делать больше меньшим кодом

Документ обобщает **техническую документацию** (DONE_01–04 и канон), оценивает **что сделано и что нет** и предлагает **как расширять дальше**, чтобы писать меньше кода и реализовывать только нужное.

**Для кого:** для вас при изучении документации и планировании следующих шагов.

---

## 1. Что уже сделано (сводка)

| Уровень | Документ | Статус | Основное |
|--------|----------|--------|----------|
| **Foundation** | DONE_01 | ✅ | Block, Port, Node, Edge, Graph (структура), Registry, внешние I/O, get_input_spec / get_output_spec |
| **Task nodes + Graph** | DONE_02 | ✅ | Роли, role_rules, абстрактные узлы-задачи, add_node(block/type, auto_connect), add_edge, to_config/from_config, state_dict, save_config, save_checkpoint, from_yaml, from_template, validate, trainable_parameters |
| **Graph engine** | DONE_03 | ✅ | Исполнитель: run(graph, inputs), топология, SCC/циклы, буфер, callbacks, dry_run, опциональные порты; save/load dir; infer_exposed_ports; backend= в checkpoint |
| **Pipeline** | DONE_04 | ✅ | Pipeline, PipelineEdge; add_graph, add_edge, expose I/O; validate (DAG, достижимость); run_pipeline; to_config/from_config, save/load, checkpoints; trainable; infer_exposed_ports |

**Единый контракт уже есть:** и Graph, и Pipeline предоставляют `get_input_spec()`, `get_output_spec()`, `run(inputs) -> outputs`. Stage и World будут использовать тот же контракт (Stage = run(pipeline, …), World = run(stage, …)).

---

## 2. Чего ещё нет (пробелы по уровням)

### 2.1 Foundation / Graph (TODO_01–03)

- **infer_exposed_ports** — в PLAN_TODO_03 помечен как «опционально»; фактически **сделан** в графе (DONE_03).
- **Checkpoint backend** — пока только `"json"`; `"torch"` / `"safetensors"` — заглушки на случай блоков с тензорами (TODO_07 или при добавлении PyTorch).
- **Config ref** — конфиг узла графа может быть `{ "ref": "path/to.yaml" }`; **реализовано** через `_resolve_config_ref`.
- Критичных пробелов по Graph нет; он готов для Pipeline и Stage.

### 2.2 Pipeline (TODO_04)

| Требование канона | Сделано? | Примечание |
|-------------------|----------|------------|
| Структура, add_graph, add_edge, expose I/O | ✅ | |
| Валидация DAG, достижимость | ✅ | |
| run(pipeline, inputs), буфер, топология | ✅ | |
| to_config, from_config, save/load, checkpoints (dir на узел) | ✅ | |
| trainable_parameters, set_trainable | ✅ | |
| infer_exposed_ports | ✅ | |
| **Конфиг: ref для конфига графа** | ⏸ | Поддерживается вложенный конфиг графа; загрузка по `ref: path` в конфиге узла пайплайна не подключена. |
| **save_checkpoint format="archive"** | ⏸ | Реализован только format="dir"; архив (например zip) — нет. |
| **Pipeline.from_yaml(path)** | ✅ | Добавлен: класс-метод загружает YAML/JSON и вызывает from_config. |
| **Pipeline.from_template(name, **kwargs)** | ⏸ | Не реализован. |
| **dry_run** для пайплайна | ⏸ | run(..., dry_run=True) не реализован. |
| **Multi-endpoint (endpoint_url на узел)** | ⏸ | Не реализовано. |
| **Кэширование топологического порядка** | ✅ | Порядок считается при каждом run; можно кэшировать и инвалидировать при add_graph/add_edge. |

Итого: **ядро пайплайна готово**. Не сделано: ref в конфиге узла пайплайна, архивный checkpoint, from_template, dry_run, multi-endpoint.

### 2.3 Stage (TODO_05) — не начат

- **Stage** = граф **пайплайнов** (тот же паттерн: Pipeline = граф графов).
- **Отличие:** контракт в терминах **state** (входные/выходные блоки state), **execution_condition(state)**, маппинг **state ↔ порты пайплайна**.
- Нужно: класс Stage, add_pipeline, add_edge, set_state_contract, set_execution_condition, run(stage, state, action?) → state; validate; to_config/from_config, save/load.

### 2.4 World (TODO_06) — не начат

- **World** = контейнер **этапов (Stages)** с **циклом**; state передаётся по циклу; **storage** для state при «World update»; **initial_world**; **execution_condition** на этап.
- Нужно: класс World, add_stage, set_cycle, set_state_schema, set_storage, run(world, state, action?) → state; save/load.

### 2.5 Будущее (TODO_07)

- **Модели через API (обязательно):** поддержка облачных и API-провайдеров на уровне **любого блока**, для которого у провайдера есть API — LLM, VLM, эмбеддинги, генерация изображений по API и т.д., без скачивания и загрузки весов. Канон: [WorldGenerator_2.0/LLM_API_SUPPORT.md](../WorldGenerator_2.0/LLM_API_SUPPORT.md).
- **Повторное использование модели:** один чекпоинт → один экземпляр; канон: [WorldGenerator_2.0/MODEL_REUSE.md](../WorldGenerator_2.0/MODEL_REUSE.md).
- **Обучение при повторном использовании и при API:** коллизии, LoRA при общем блоке, обучение мира с API-узлами (промпт-адаптеры), сценарии и обходы — канон: [WorldGenerator_2.0/TRAINING_REUSE_AND_API_SCENARIOS.md](../WorldGenerator_2.0/TRAINING_REUSE_AND_API_SCENARIOS.md).
- Модальности (VLM, видео, аудио), бэкенды (ONNX, vLLM), бэкенды checkpoint (torch, safetensors), async run, батчинг, наблюдаемость, JSON Schema для конфигов. Всё — «по мере необходимости».

---

## 3. Как расширять меньшим кодом — идеи проектирования

### 3.1 Один и тот же паттерн на каждом уровне

На каждом уровне есть: **узлы** (id → дочерний объект), **рёбра** по портам из get_output_spec/get_input_spec дочернего, **внешние I/O**, **валидация** (DAG, порты), **run** (топологический порядок, буфер, вызов child.run), **сериализация** (to_config/from_config, checkpoints = объединение дочерних). Pipeline и Stage — один и тот же алгоритм; меняется только тип дочернего объекта (Graph или Pipeline). World чуть иначе (цикл вместо DAG), но исполнитель остаётся простым.

**Вывод:** один раз реализовать обобщённый «граф исполняемых объектов», параметризованный типом дочернего объекта, способом создания из конфига, способом run и save/load. Тогда Pipeline и Stage становятся тонкими обёртками.

### 3.2 Что реализовать в первую очередь

1. **Pipeline.from_yaml(path)** — сделано.
2. **Stage (TODO_05)** — тот же builder-паттерн, что и Pipeline; маппинг state; execution_condition.
3. **World (TODO_06)** — цикл + state_schema + storage; run по циклу с проверкой условий.
4. **Опционально:** вынести общую логику run «граф исполняемых» для Pipeline и Stage.
5. **Отложить:** ref в конфиге узла пайплайна, format="archive", from_template, dry_run, multi-endpoint; TODO_07 по мере надобности.

---

## 4. Конкретные следующие шаги (приоритет)

| Приоритет | Задача | Цель |
|-----------|--------|------|
| 1 | ~~Добавить Pipeline.from_yaml(path)~~ | **Сделано.** |
| 2 | **Реализовать TODO_05 (Stage)** | Разблокировать World; тот же паттерн «граф исполняемых», что и у Pipeline. |
| 3 | **Реализовать TODO_06 (World)** | Цикл + state + storage; завершает иерархию. |
| 4 | (Опционально) **Вынести общую логику run «граф исполняемых»** | Меньше дублирования между Pipeline и Stage. |
| 5 | (Позже) **Pipeline: ref, from_template, dry_run, multi-endpoint** | Когда понадобятся шаблоны, проверка без выполнения или удалённые узлы. |
| 6 | **Модели через API (без локальных весов)** | **Обязательно.** Любой блок, для которого у провайдера есть API. Канон: WorldGenerator_2.0/LLM_API_SUPPORT.md. |
| 7 | **Повторное использование модели (без дублирования в памяти)** | Один чекпоинт → один экземпляр; для всех типов моделей и всех уровней иерархии. Канон: WorldGenerator_2.0/MODEL_REUSE.md. |
| 8 | **Обучение при повторном использовании и при API** | Коллизии, LoRA при общем блоке, обучение мира с API-узлами. Канон: WorldGenerator_2.0/TRAINING_REUSE_AND_API_SCENARIOS.md. |

---

## 5. Ссылки на файлы

- **Сделано:** docs/DONE_01_FOUNDATION.md, DONE_02_TASK_NODES_AND_GRAPH.md, DONE_03_GRAPH_ENGINE.md, DONE_04_PIPELINE.md.
- **Планы:** docs/PLAN_TODO_02_AND_FORWARD.md, PLAN_TODO_03_VERIFIED_AND_FORWARD.md, OUTLINE_TODO_02_TO_07.md.
- **Канон:** WorldGenerator_2.0/TODO_04_PIPELINE.md, TODO_05_STAGE.md, TODO_06_WORLD.md, **LLM_API_SUPPORT.md**, **MODEL_REUSE.md**, **TRAINING_REUSE_AND_API_SCENARIOS.md**, Pipeline_Level.md, Stage_Level.md, World_Level.md, Scheme.md.
- **Визия и дорожная карта документации:** [docs/VISION_AND_DOCUMENTATION_ROADMAP.md](VISION_AND_DOCUMENTATION_ROADMAP.md) — что добавить, чтобы фреймворк стал золотым стандартом для диффузии и генерации миров и самым удобным в использовании.

---

## 6. Краткая сводка

- **Сделано:** Foundation → Task nodes → Graph engine → Pipeline. Контракт (get_input_spec, get_output_spec, run) единый; Graph и Pipeline готовы к Stage и World.
- **Не сделано:** доработки Pipeline (ref, archive, from_template, dry_run, multi-endpoint); **Stage (TODO_05)**; **World (TODO_06)**; **модели через API** (любой блок, когда у провайдера есть API); **повторное использование модели** (один чекпоинт → один экземпляр); **обучение при reuse и API** (см. TRAINING_REUSE_AND_API_SCENARIOS); TODO_07 по мере надобности.
- **Чтобы расширять меньшим кодом:** (1) Pipeline.from_yaml уже добавлен. (2) Реализовать Stage, переиспользуя паттерн «граф исполняемых». (3) Реализовать World как цикл + state + storage. (4) Отложить опциональные фичи Pipeline и TODO_07 до появления потребности.
