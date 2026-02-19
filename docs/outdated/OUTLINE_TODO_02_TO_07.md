# Outline: TODO_02 – TODO_07 (WorldGenerator_2.0)

**Канон проекта:** [CANON.md](../WorldGenerator_2.0/CANON.md) — единый всеобъемлющий канон (WorldGenerator_2.0/).

Краткий конспект по канону WorldGenerator_2.0 для дальнейшей разработки движка. Источники: папка [WorldGenerator_2.0](../WorldGenerator_2.0/) — все [TODO_01](../WorldGenerator_2.0/TODO_01_FOUNDATION.md)…[TODO_07](../WorldGenerator_2.0/TODO_07_FUTURE_AND_IMPROVEMENTS.md), [Abstract_Task_Nodes](../WorldGenerator_2.0/Abstract_Task_Nodes.md), [Graph_Level](../WorldGenerator_2.0/Graph_Level.md), [Scheme](../WorldGenerator_2.0/Scheme.md), [AGENT_SYSTEMS_SUPPORT](../WorldGenerator_2.0/AGENT_SYSTEMS_SUPPORT.md), [EXPANSION_UNIVERSE_GRAPH_OF_WORLDS](../WorldGenerator_2.0/EXPANSION_UNIVERSE_GRAPH_OF_WORLDS.md) и др.

**Расширенный канон:** агентные системы; полная гибкость. **Гиперграф:** вся архитектура — гиперграф; крупномасштабные гиперссылки для контекстов и взаимодействия ([HYPERGRAPH_AND_HYPERLINKS.md](../WorldGenerator_2.0/HYPERGRAPH_AND_HYPERLINKS.md)). **End Worlds:** фиксированные миры, к которым можно привязываться без их изменения ([END_WORLDS.md](../WorldGenerator_2.0/END_WORLDS.md)). Уровень Вселенной: гиперграф миров, этап для сообщества, кросс-мировое влияние, End Worlds ([EXPANSION_UNIVERSE_GRAPH_OF_WORLDS.md](../WorldGenerator_2.0/EXPANSION_UNIVERSE_GRAPH_OF_WORLDS.md)). Роли переходов (идеальные задачи): [Scheme.md](../WorldGenerator_2.0/Scheme.md) §1.1 (переход = метаморфоза, §1.0).

---

## TODO_02 — Узлы-задачи и уровень графа

- **Часть A — Abstract Task Nodes:** двойное наследование (Block + роль). Типы: Backbone, Solver, Codec, Conditioner, Tokenizer, Adapter, Guidance (+ опционально NoiseSchedule, PositionEmbedder). У каждого: контракт портов, семантика выполнения, типичные связи.
- **Таблица role rules:** (роль_источника, роль_приёмника) → какие порты с какими соединять. Нужна для **AddNode(..., auto_connect=True)**.
- **Часть B — Граф:** один граф = одна задача. **AddNode(block_or_type, node_id=None, config=None, auto_connect=True)** — блок или тип из реестра; при auto_connect создаются рёбра по role rules. **AddEdge** с валидацией. Внешние входы/выходы графа (expose_input/expose_output — уже есть в фундаменте).
- **Исполнитель:** run(graph, inputs) → outputs; топология, циклы (backbone–solver), буфер по (node_id, port_name).
- **Сериализация:** конфиг (узлы, рёбра, I/O) + чекпоинт весов; from_config, from_template. Обучаемость: выбор trainable узлов, trainable_parameters().

---

## TODO_03 — Графовый движок

- **Структура:** узлы, рёбра, exposed_inputs/exposed_outputs (уже в фундаменте), индексы in_edges/out_edges (уже есть).
- **Builder:** AddNode с авто-связыванием по role rules (требует TODO_02), AddEdge, expose_input/expose_output.
- **Валидация:** при AddEdge; полная validate() — достижимость, обязательные порты, циклы (начальные значения снаружи).
- **Исполнитель:** топологическая сортировка, SCC, циклы с N итерациями; буфер (node_id, port_name); **run(graph, inputs, training=..., num_loop_steps=...)** → outputs; callbacks, device.
- **Сериализация:** schema_version, save_config, save_checkpoint, load_from_checkpoint; один файл с префиксами node_id или каталог по узлам.
- **Конфиг и шаблоны:** from_config/from_yaml, from_template(name, **kwargs). get_input_spec/get_output_spec для пайплайна и multi-endpoint.

---

## TODO_04 — Пайплайн

- Узлы = **графы** (целые графы). Рёбра соединяют **внешние** порты графов.
- AddGraph(graph_or_config), AddEdge между внешними портами графов. expose_input/expose_output пайплайна. run(pipeline, inputs) → вызов run(graph, …) по топологии.
- Сериализация: конфиг пайплайна (графы, рёбра, I/O) + чекпоинты графов.

---

## TODO_05 — Этап (Stage)

- Узлы = **пайплайны**. Рёбра по внешним портам пайплайнов. Контракт по **state** (входные/выходные блоки state).
- **State schema** (в каноне — пять блоков). Условия выполнения этапа по заполненности state (Scheme: Философ при 1,2,3; Автор при 1–4; World update при всех 5; Development безусловно).
- run(stage, state) с маппингом state ↔ входы/выходы пайплайнов.

---

## TODO_06 — Мир (World)

- Узлы = **этапы**. **Цикл** (порядок: Философ → Автор → Среда → Архитектор → Творец). По рёбрам передаётся **state**.
- **State_schema**, **storage** для сохранения state при World update. **Initial_world**, **Action** (опционально). Условия выполнения по Scheme.
- run(world, state, action?) — последовательный вызов этапов по циклу с проверкой условий.

---

## TODO_07 — Дальнейшее развитие

- Модальности: VLM, видео, аудио, мультимодальность, стриминг.
- Бэкенды: ONNX, vLLM, TGI, квантизация. Async run, батчинг, распределённое обучение.
- Наблюдаемость: трейсинг, метрики, профилирование. Валидация конфигов (JSON Schema), контрактные тесты.

---

## Связь с фундаментом (TODO_01)

- Блок, узел, порты, реестр — основа для всех уровней.
- Граф уже поддерживает: exposed_inputs/exposed_outputs, get_input_spec/get_output_spec, schema_version в конфиге, индексы рёбер. Это задел для TODO_03 (исполнитель и контракт графа).

Дальше: реализация **TODO_02** (узлы-задачи + роль + role rules), затем **TODO_03** (исполнитель графа и валидация).
