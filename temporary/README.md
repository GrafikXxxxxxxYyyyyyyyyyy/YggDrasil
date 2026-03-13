# Временная папка разработки (temporary)

**Назначение:** всё, что относится к **процессу разработки** проекта Иггдрасиль и построению кодовой базы фреймворка: планирование, TODO, лог разработки, что сделано и что нужно сделать.

**Важно:** папка **временная**. После завершения разработки фреймворка будет удалена. В ней не хранится каноническая документация (она в `documentation/`) и не живёт стабильный код (он будет в корне репозитория или в выделенной директории).

---

## Содержимое

| Файл | Назначение |
|------|------------|
| **README.md** (этот файл) | Описание папки и её содержимого. |
| **TODO.md** | Текущие задачи: что в работе, что в очереди, приоритеты. |
| **DEV_LOG.md** | Лог разработки: что сделано, когда, краткое описание изменений. |
| **IMPLEMENTATION_PLAN.md** | Детальный план реализации: что программировать, в каком порядке, с привязкой к канону (documentation/docs/). |
| **PHASE_0_STRUCTURE.md** | Полная спецификация фазы 0: структура репозитория, папка tests/ в корне, пакет yggdrasill/ в корне, pyproject.toml, setup, requirements, чек-лист выполнения. |
| **PHASE_1_FOUNDATION.md** | Полный технический план фазы 1: Port, Block, Node, Registry — контракты, API, тесты, порядок реализации, приёмочные критерии. |
| **PHASE_2_ENGINE.md** | Полный технический план фазы 2: гиперграфовый движок — Edge, Hypergraph, Validator, Planner, Executor, run(hypergraph, inputs)→outputs, тесты, приёмочные критерии. |
| **PHASE_3_TASK_HYPERGRAPH.md** | Полный технический план фазы 3: гиперграф задачи — add_node(block_type, config), from_config, to_config, run, get_input_spec/get_output_spec, infer_exposed_ports, state_dict/load_state_dict, тесты, приёмочные критерии. |
| **PHASE_4_ABSTRACT_TASK_NODES.md** | Полный технический план фазы 4: семь ролей узлов-задач (Backbone, Injector, Conjector, Inner Module, Outer Module, Helper, Converter), абстракции, заглушки role/identity, регистрация, role_from_block_type, опционально автосвязывание, тесты. |
| **PHASE_5_SERIALIZATION.md** | Полный технический план фазы 5: сериализация блока и гиперграфа задачи — конфиг + чекпоинт, save_block/load_block, save/save_config/save_checkpoint, load/load_config/load_from_checkpoint, дедупликация по block_id, форматы файлов, версионирование, воспроизводимость, тесты. |
| **PHASE_6_WORKFLOW.md** | Полный технический план фазы 6: воркфлоу как гиперграф гиперграфов (узлы = гиперграфы задач), тот же движок (Validator, Planner, Executor), Workflow (add_node, add_edge, expose_*, from_config, to_config, run, state_dict, save/load), циклы между гиперграфами, валидация, сериализация, тесты. |
| **PHASE_7_STAGE.md** | Полный технический план фазы 7 (экспериментальная): стадия как гиперграф воркфлоу (узлы = воркфлоу), State как объект с атрибутами + директория мира, три базовых типа стадий (Автор/Среда/Творец), state_input_map/state_output_map, can_run, run(stage, state, context)→state, взаимодействие агентов с файловой системой. |
| **PHASE_8_WORLD.md** | Полный технический план фазы 8 (экспериментальная): мир как гиперграф стадий (Автор/Среда/Творец), цикл мира, связь State и директории мира, исполняемая часть и контент, развивающиеся миры и End World, контракт run(world, state, action, context)→state. |
| **PHASE_9_UNIVERSE.md** | Полный технический план фазы 9 (экспериментальная): вселенная как гиперграф миров, worlds/edges/ether/order, payload_spec и переход сущностей, контракт run(universe, world_inputs, **options)→results, сериализация вселенной и граница онтологии. |
| **PSEUDOCODE_ARCHITECTURE.md** | Псевдокод архитектуры: как два начала (Block, Node), узлы-задачи (двойное наследование) и гиперграф без обёртки выглядят в коде; примеры AbstractBaseBlock, AbstractGraphNode, AbstractBackbone, IdentityBackbone, add_node, run. |
| **PHASE_10_SERIALIZATION_POLISH.md** | Наметки для фазы 10 (экспериментальная): единый контракт сериализации для всех уровней, воспроизводимость и интеграционные тесты, версионирование схем, практические стратегии хранения (FS/БД) без изменения онтологии. |

---

## Как пользоваться

- **Планирование** — опираться на [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) и документацию в `documentation/` (Scheme.md, docs/01–07, HYPERGRAPH_ENGINE, SERIALIZATION и т.д.).
- **Текущая работа** — вести [TODO.md](TODO.md), отмечать выполненное в [DEV_LOG.md](DEV_LOG.md).
- По завершении фреймворка — удалить папку `temporary/` или архивировать её вне репозитория.
