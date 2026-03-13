# TODO — Часть 1: Движок и поддержка диффузии / язык / агенты

Детальные задачи по **Части 1** плана реализации. Отмечать выполненное и переносить в [DEV_LOG.md](DEV_LOG.md). Связь с планом: [WORK_PLAN.md](WORK_PLAN.md), [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

---

## Фаза 0. Структура репозитория

**Документ:** [PHASE_0_STRUCTURE.md](PHASE_0_STRUCTURE.md).

### Структура каталогов

- [ ] Создать пакет `yggdrasill/` в корне с подпакетами: `foundation`, `task_nodes`, `engine`, `hypergraph`, `workflow`, `stage`, `world`, `universe`.
- [ ] В каждом подпакете — `__init__.py` (пустой или с docstring).
- [ ] В корне создать папку `tests/` с подпапками: `foundation`, `task_nodes`, `engine`, `hypergraph`, `workflow`; `conftest.py` при необходимости.
- [ ] В `yggdrasill/__init__.py` задать `__version__` (например `"0.1.0"`).

### Конфигурация сборки и зависимости

- [ ] Добавить `pyproject.toml`: метаданные пакета, зависимости (runtime), optional `[dev]` (pytest, ruff, и т.д.), setuptools, точка входа при необходимости.
- [ ] При необходимости: `requirements.txt` / `requirements-dev.txt` или ссылка на optional-dependencies.
- [ ] Убедиться, что `pip install -e .` и `pip install -e ".[dev]"` выполняются без ошибок.

### Запуск тестов

- [ ] Запуск из корня: `pytest tests/` работает (пустые тесты или один smoke test).
- [ ] Структура готова к добавлению тестов по фазам 1–6.

### Чек-лист приёмки фазы 0

- [ ] Импорт: `from yggdrasill import __version__` (или `import yggdrasill`) возможен после `pip install -e .`.
- [ ] Все пункты из PHASE_0_STRUCTURE выполнены.

---

## Фаза 1. Фундамент (Block, Node, Port, Registry)

**Документ:** [PHASE_1_FOUNDATION.md](PHASE_1_FOUNDATION.md).

### Port

- [ ] Класс/тип Port: имя, направление (IN/OUT), тип или схема, опциональность (обязательный/опциональный).
- [ ] Опционально: политика агрегации (concat, sum, first) для входа с несколькими рёбрами.
- [ ] Метод/функция совместимости портов (compatible_with или аналог).

### AbstractBaseBlock

- [ ] Атрибуты: `block_id`, `block_type`, `config` (копия или неизменяемый вид).
- [ ] Методы: `forward(inputs) -> outputs` (абстрактный или переопределяемый), без объявления портов в блоке (порты у узла).
- [ ] `state_dict() -> dict`, `load_state_dict(state_dict)` (для блока без весов — пустой dict / no-op).
- [ ] Не содержит методов declare_ports, get_input_ports, get_output_ports (это у AbstractGraphNode).

### AbstractGraphNode

- [ ] Наследуется от ABC; не хранит отдельный блок; предназначен только для наследования вместе с AbstractBaseBlock.
- [ ] Атрибут: `node_id`.
- [ ] Абстрактный метод `declare_ports()`; производные `get_input_ports()`, `get_output_ports()`.
- [ ] `run(inputs) -> outputs`: делегирует в `self.forward(inputs)` (узлы-задачи реализуют forward в одном классе с declare_ports).
- [ ] Конструктор: `__init__(self, node_id: str)` (и при наследовании с Block — вызов обоих родительских __init__).

### BlockRegistry

- [ ] `register(block_type, factory)` — регистрация фабрики (класс или функция) по строковому типу.
- [ ] `build(config) -> block` — создание экземпляра по config (config содержит block_type и параметры); для узлов-задач возвращается один объект Block+Node.
- [ ] Глобальный или получаемый реестр; возможность регистрации узлов-задач по block_type.

### Тесты фазы 1

- [ ] Тесты Port: создание, направление, совместимость.
- [ ] Тесты минимального блока (без весов): state_dict, load_state_dict.
- [ ] Тесты минимального узла-задачи (двойное наследование): declare_ports, get_input_ports, get_output_ports, run → forward.
- [ ] Тесты реестра: register, build по block_type.

### Чек-лист приёмки фазы 1

- [ ] Узел-задача (один класс, наследующий AbstractBaseBlock и AbstractGraphNode) создаётся через registry.build(config), имеет declare_ports и run(inputs) → forward(inputs).
- [ ] Два начала не смешаны: порты и положение в графе — у Node; данные и вычисления — у Block.

---

## Фаза 2. Гиперграфовый движок

**Документ:** [PHASE_2_ENGINE.md](PHASE_2_ENGINE.md).

### Edge и хранилище структуры

- [ ] Тип Edge: source_node, source_port, target_node, target_port (арность 2).
- [ ] Хранилище (Hypergraph или аналог): узлы (node_id → узел), список рёбер, exposed_inputs, exposed_outputs, execution_version.
- [ ] Методы: add_node(node_id, node), add_edge(edge), expose_input(…), expose_output(…), get_node(id), get_edges(), get_edges_in(id), get_edges_out(id), get_input_spec(), get_output_spec().
- [ ] При изменении структуры — инкремент execution_version.

### Validator

- [ ] Вход: структура (Hypergraph или объект с node_ids, get_node, get_edges, get_input_spec, get_output_spec).
- [ ] Проверки: все node_id и port_name в рёбрах существуют; типы портов совместимы; обязательные входы покрыты.
- [ ] Для графа с циклом: проверка наличия семантики (num_loop_steps или graph_kind).
- [ ] Результат: ValidationResult (errors, warnings); при строгом режиме — исключение или флаг.

### Execution Planner

- [ ] Вход: структура графа (узлы, рёбра).
- [ ] DAG: топологическая сортировка → порядок вызовов.
- [ ] Граф с циклом: разбиение на фазы (начальная, итеративная «K раз», конечная); K из metadata или options.
- [ ] Кэш плана по execution_version; инвалидация при изменении графа.

### Edge Buffers

- [ ] Ключ буфера: (node_id, port_name).
- [ ] Инициализация из внешних inputs (exposed_inputs).
- [ ] После run узла: запись выходов в буферы приёмников; при нескольких рёбрах на один порт — агрегация по политике порта.
- [ ] Сбор внешних выходов по exposed_outputs в словарь outputs.

### Executor

- [ ] Сигнатура: run(structure, inputs, **options) -> outputs.
- [ ] Алгоритм: валидация (опционально) → план → инициализация буферов → обход по плану (сбор входов узла из буферов, node.run(inputs), запись выходов в буферы) → сбор outputs.
- [ ] Опции: num_loop_steps, validate, device, dry_run, callbacks и т.д.

### Тесты фазы 2

- [ ] Тесты Edge и хранилища: add_node, add_edge, expose_*, get_*.
- [ ] Тесты Validator: валидный граф, отсутствующий узел/порт, несовместимые типы, непокрытые обязательные входы.
- [ ] Тесты Planner: DAG — порядок; цикл из двух узлов — фазы и K шагов.
- [ ] Тесты Executor: цепочка 2–3 узлов (заглушки), run(hypergraph, inputs) → ожидаемые outputs; цикл K шагов.

### Чек-лист приёмки фазы 2

- [ ] run(hypergraph, inputs, num_loop_steps=K) выполняется для цепочки и для одного цикла; outputs соответствуют exposed_outputs.

---

## Фаза 3. Гиперграф задачи

**Документ:** [PHASE_3_TASK_HYPERGRAPH.md](PHASE_3_TASK_HYPERGRAPH.md).

### API гиперграфа задачи

- [ ] add_node(node_id, block_type, config, registry?, block_id?, trainable?, …) — создание узла через реестр, добавление в граф (один объект Block+Node).
- [ ] add_node(node_id, node) — добавление готового узла.
- [ ] add_edge(source_node, source_port, target_node, target_port), remove_edge, remove_node при необходимости.
- [ ] expose_input(…), expose_output(…).
- [ ] from_config(config, registry?, validate?) — построение гиперграфа из словаря.
- [ ] to_config() — экспорт в словарь (nodes, edges, exposed_*, graph_id, graph_kind, metadata, schema_version).
- [ ] graph_id, graph_kind, metadata (в т.ч. num_loop_steps).

### Контракт для движка и воркфлоу

- [ ] get_input_spec(include_dtype?), get_output_spec(include_dtype?) — по exposed_inputs/exposed_outputs и портам узлов.
- [ ] run(inputs, **options) — делегирование в engine.run(self, inputs, **options).
- [ ] to(device), trainable_parameters(), set_trainable(node_id, bool).
- [ ] state_dict(), load_state_dict(state) — агрегат по узлам (по node_id; при общем block_id — дедупликация в фазе 5).

### Опционально

- [ ] infer_exposed_ports() — вывод внешних входов/выходов по отсутствующим входящим/исходящим рёбрам.
- [ ] add_hyperedge(ends), auto_connect — заглушка или базовая логика.

### Тесты фазы 3

- [ ] Сборка гиперграфа из конфига (from_config → to_config roundtrip).
- [ ] run(hypergraph, inputs) с цепочкой и с циклом (K шагов).
- [ ] get_input_spec, get_output_spec совпадают с ожидаемыми портами.

### Чек-лист приёмки фазы 3

- [ ] Гиперграф собирается из конфига, прогоняется через движок, экспортируется в конфиг; может использоваться как «узел» воркфлоу (get_input_spec, get_output_spec, run).

---

## Фаза 4. Абстрактные узлы-задачи (семь ролей)

**Документ:** [PHASE_4_ABSTRACT_TASK_NODES.md](PHASE_4_ABSTRACT_TASK_NODES.md).

### Абстрактные классы ролей

- [ ] AbstractBackbone (порты по канону 02).
- [ ] AbstractInjector, AbstractConjector, AbstractInnerModule, AbstractOuterModule, AbstractHelper, AbstractConverter.
- [ ] Каждый — наследник AbstractBaseBlock и AbstractGraphNode; declare_ports() по документу 02; абстрактный forward().

### Константы и роль по типу

- [ ] Константы ролей (backbone, injector, conjector, inner_module, outer_module, helper, converter).
- [ ] role_from_block_type(block_type) -> роль.

### Заглушки и регистрация

- [ ] Заглушка на каждую роль (identity/passthrough): регистрация как `backbone/identity`, `conjector/identity` и т.д.
- [ ] register_all_stubs() или автоматическая регистрация при импорте пакета task_nodes.

### Опционально

- [ ] Правила типичных связей между ролями (role_rules); suggest_edges, apply_auto_connect.

### Тесты фазы 4

- [ ] Сборка гиперграфа из конфига с block_type узлов-задач (например backbone/identity, converter/identity); run, проверка выходов.
- [ ] Тесты каждой заглушки: порты и run.
- [ ] role_from_block_type для всех зарегистрированных типов.
- [ ] Цепочка Converter → Backbone → Converter и цикл Backbone ↔ Inner Module (заглушки).

### Чек-лист приёмки фазы 4

- [ ] Семь ролей объявлены и доступны через реестр; гиперграф из конфига с ролями собирается и выполняется.

---

## Фаза 5. Сериализация блока и гиперграфа задачи

**Документ:** [PHASE_5_SERIALIZATION.md](PHASE_5_SERIALIZATION.md).

### Блок

- [ ] get_config() у блока (или хелпер): block_type, block_id, config.
- [ ] save_block(block, path_or_dir), load_block(path_or_dir, registry?) — конфиг + чекпоинт (state_dict).
- [ ] Формат файлов конфига и чекпоинта зафиксирован (JSON/YAML для конфига; .pkl/.pt или аналог для state_dict).

### Гиперграф задачи

- [ ] save(path), save_config(path), save_checkpoint(path).
- [ ] load(path, registry?), load_config(path), load_from_checkpoint(path).
- [ ] Дедупликация по block_id при сохранении/загрузке чекпоинта (один state_dict на block_id).
- [ ] schema_version в конфиге.

### Тесты фазы 5

- [ ] Roundtrip блока: save_block → load_block → тот же state_dict (или эквивалентное поведение forward).
- [ ] Roundtrip гиперграфа: save → load → to_config совпадает с исходным; run после load даёт тот же результат при тех же входах.
- [ ] Общий block_id у двух узлов — в чекпоинте одна запись по block_id.

### Чек-лист приёмки фазы 5

- [ ] Гиперграф задачи сохраняется и загружается; воспроизводимость run при одинаковых конфиге, чекпоинте и входах.

---

## Фаза 6. Воркфлоу

**Документ:** [PHASE_6_WORKFLOW.md](PHASE_6_WORKFLOW.md).

### Класс Workflow

- [ ] _nodes: graph_id → Hypergraph; _edges (Edge из engine); _exposed_inputs, _exposed_outputs; _execution_version.
- [ ] add_node(graph_id, hypergraph), add_node(graph_id, config, registry?); add_edge(source_graph_id, source_port, target_graph_id, target_port).
- [ ] expose_input(…), expose_output(…).
- [ ] get_node(graph_id), get_edges(), get_input_spec(), get_output_spec(), get_edges_in/out, node_ids — контракт для движка.
- [ ] run(inputs, **options) — делегирование в engine.run(self, inputs, **options).
- [ ] from_config(config, registry?), to_config().
- [ ] state_dict(), load_state_dict(state); to(device), trainable_parameters(), set_trainable(graph_id, bool).

### Движок для воркфлоу

- [ ] Validator, Planner, Executor работают с Workflow как со структурой (get_node возвращает Hypergraph с run, get_input_spec, get_output_spec).
- [ ] Циклы между гиперграфами: итеративная фаза K раз (num_loop_steps / num_workflow_steps).

### Сериализация воркфлоу

- [ ] save(path), save_config(path), save_checkpoint(path); load(path, registry?), load_config(path), load_from_checkpoint(path).
- [ ] Конфиг воркфлоу + чекпоинты гиперграфов (по graph_id); при общем ref — один экземпляр, один чекпоинт.

### Опционально

- [ ] infer_exposed_ports(); auto_connect между гиперграфами (suggest_auto_edges, apply_auto_connect).

### Тесты фазы 6

- [ ] Roundtrip to_config/from_config воркфлоу.
- [ ] run(workflow, inputs) — цепочка из 2–3 гиперграфов; проверка выходов.
- [ ] Цикл между гиперграфами (K шагов).
- [ ] save/load воркфлоу; после load run даёт тот же результат при тех же входах.
- [ ] Дедупликация при общем ref (один гиперграф на два graph_id или shared ref).

### Чек-лист приёмки фазы 6

- [ ] Воркфлоу собирается, выполняется движком, сохраняется и загружается; циклы между гиперграфами поддерживаются.

---

## Поддержка доменов: диффузия, язык, агенты

Цель: убедиться, что архитектура и контракты позволяют реализовать типовые сценарии; при необходимости добавить минимальные примеры или тесты.

### Диффузионные системы

- [ ] Сценарий задокументирован или покрыт тестом: Backbone (ядро генерации) + Conjector (например условие/текст) + при необходимости Injector/Inner Module; порты и рёбра по канону 02.
- [ ] Заглушки или минимальные блоки позволяют собрать граф «диффузия» и прогнать run (хотя бы с фиктивными тензорами/структурами).
- [ ] При наличии реального блока — один интеграционный тест или пример в документации.

### Языковые системы

- [ ] Сценарий: Backbone (LLM/генерация текста) + Converter (форматирование/парсинг) или Helper (вспомогательные шаги); цепочка или цикл.
- [ ] Заглушки позволяют собрать граф «языковая цепочка» и прогнать run.
- [ ] Контракт портов (текст, структуры, токены) не противоречит канону; при необходимости опциональные порты и агрегация.

### Агентные системы

- [ ] Если agent_loop в scope Части 1: узел может возвращать tool_calls; движок или обвязка выполняет инструменты и передаёт tool_results обратно в узел; цикл до отсутствия tool_calls или max_steps.
- [ ] Если agent_loop отложен: контракт (выход с tool_calls) и место в архитектуре задокументированы; заглушка узла с tool_calls не ломает run (результат игнорируется или передаётся дальше).
- [ ] Мини-тест или пример: граф с одним «агентным» узлом (заглушка), run завершается и возвращает выход.

### Сводка по доменам

- [ ] В README или docs кратко описано: как подключить блок диффузии/LLM/агента через реестр и как собрать гиперграф под соответствующий сценарий.
- [ ] Все три домена (диффузия, язык, агенты) либо покрыты тестом/примером, либо явно помечены как «достаточно контрактов и заглушек, конкретные блоки — отдельно».

---

## Покрытие кейсов и интеграционные тесты

### Граф и движок

- [ ] DAG: линейная цепочка (2–5 узлов); run → порядок вызовов и выходы.
- [ ] DAG: разветвление и слияние (несколько рёбер в один порт — агрегация).
- [ ] Один цикл в гиперграфе: два узла, K итераций; выходы после K шагов.
- [ ] Опциональные порты: узел с опциональным входом; ребро отсутствует — значение по умолчанию или пропуск.
- [ ] Внешние входы/выходы: expose_input, expose_output; run(hypergraph, inputs) с ключами по именам; outputs содержат ожидаемые ключи.

### Воркфлоу

- [ ] Цепочка гиперграфов: G1 → G2 → G3; run(workflow, inputs) → выход G3.
- [ ] Цикл в воркфлоу: G1 → G2 → G1 (K шагов); выходы после K итераций.
- [ ] Exposed inputs/outputs воркфлоу; run(workflow, inputs) с ключами по graph_id и port или по name.

### Сериализация и воспроизводимость

- [ ] Гиперграф: save → load → run(inputs) даёт тот же результат, что run до save (при фиксированных inputs и чекпоинте).
- [ ] Воркфлоу: save → load → run(inputs) — то же.
- [ ] Отсутствующий чекпоинт при load: граф восстанавливается по конфигу; поведение без весов или с предупреждением задокументировано.

### Ошибки и граничные случаи

- [ ] Валидатор: несуществующий node_id в ребре → ошибка.
- [ ] Валидатор: несуществующий port_name → ошибка.
- [ ] Валидатор: непокрытый обязательный внешний вход → ошибка.
- [ ] Пустой граф или воркфлоу без узлов: явная ошибка или документированное поведение.
- [ ] run с лишними ключами в inputs / отсутствующими ключами: документированное поведение (игнор лишнего, ошибка при отсутствии обязательного).

---

## Общие задачи Части 1

- [ ] Все тесты из PHASE_0 … PHASE_6 реализованы и проходят.
- [ ] `pytest tests/` из корня — зелёный.
- [ ] Lint/format (ruff или аналог) настроен и проходит по коду (опционально, по PHASE_0).
- [ ] README корня проекта обновлён: как установить, как запустить тесты, как собрать минимальный гиперграф и воркфлоу (кратко).
- [ ] Критерии завершения Части 1 из WORK_PLAN.md выполнены; Часть 2 не начинать до закрытия Части 1.

---

## Прогресс

| Фаза | Статус | Примечание |
|------|--------|------------|
| 0 | ⬜ Не начата | |
| 1 | ⬜ Не начата | |
| 2 | ⬜ Не начата | |
| 3 | ⬜ Не начата | |
| 4 | ⬜ Не начата | |
| 5 | ⬜ Не начата | |
| 6 | ⬜ Не начата | |
| Домены + кейсы | ⬜ Не начато | |

По мере выполнения менять статус на «В работе» / «Выполнена» и переносить выполненные пункты в DEV_LOG.
