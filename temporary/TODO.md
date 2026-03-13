# TODO — текущие задачи разработки

Актуальный список задач. По мере выполнения переносить в [DEV_LOG.md](DEV_LOG.md).

**Оптимальный roadmap** зафиксирован ниже; детальные чек-листы по фазам: [TODO_PART1.md](TODO_PART1.md) (Часть 1), [TODO_PART2.md](TODO_PART2.md) (Часть 2). Полная спецификация фаз: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

---

## Оптимальный roadmap

### Часть 1: Движок и минимальный вертикальный срез

Цель — как можно раньше получить **один полный путь**: один блок → один гиперграф → один `run(hypergraph, inputs)` → выход. Дальше — циклы, роли, сериализация, воркфлоу, домены.

| Фаза | Цель | Результат фазы | Детали |
|------|------|----------------|--------|
| **0** | Структура репозитория | Пакет `yggdrasill/`, `tests/`, pyproject.toml, `pytest tests/` работает | [PHASE_0_STRUCTURE.md](PHASE_0_STRUCTURE.md) |
| **1** | Фундамент | Block, Node, Port, реестр типов; один конкретный блок (например Identity) и `block.run(inputs)` | [PHASE_1_FOUNDATION.md](PHASE_1_FOUNDATION.md), [01_FOUNDATION.md](../documentation/docs/01_FOUNDATION.md) |
| **2** | Движок | Хранилище структуры, валидатор, планировщик (топосорт для DAG), буферы, исполнитель; run по графу без циклов | [PHASE_2_ENGINE.md](PHASE_2_ENGINE.md), [HYPERGRAPH_ENGINE.md](../documentation/docs/HYPERGRAPH_ENGINE.md) |
| **3** | Гиперграф задачи | Класс гиперграфа (add_node, add_edge, expose_input/output), интеграция с движком, **первый полный run(hypergraph, inputs) → outputs**; затем циклы (num_loop_steps), опционально agent_loop | [PHASE_3_TASK_HYPERGRAPH.md](PHASE_3_TASK_HYPERGRAPH.md), [03_TASK_HYPERGRAPH.md](../documentation/docs/03_TASK_HYPERGRAPH.md) |
| **4** | Семь ролей | Backbone, Conjector, Inner Module, Outer Module, Converter, Injector, Helper — интерфейсы/базовые классы и заглушки; регистрация в реестре; сбор тестового графа из заглушек и run | [PHASE_4_ABSTRACT_TASK_NODES.md](PHASE_4_ABSTRACT_TASK_NODES.md), [02_ABSTRACT_TASK_NODES.md](../documentation/docs/02_ABSTRACT_TASK_NODES.md) |
| **5** | Сериализация | Конфиг + чекпоинт для блока и гиперграфа задачи; save/load; дедупликация по block_id | [PHASE_5_SERIALIZATION.md](PHASE_5_SERIALIZATION.md), [SERIALIZATION.md](../documentation/docs/SERIALIZATION.md) |
| **6** | Воркфлоу | Гиперграф гиперграфов; тот же движок; run(workflow, inputs); сериализация воркфлоу | [PHASE_6_WORKFLOW.md](PHASE_6_WORKFLOW.md), [04_WORKFLOW.md](../documentation/docs/04_WORKFLOW.md) |

**После фазы 6:** покрытие доменов (диффузия, LLM, агенты) по [DIFFUSION_MODELS.md](../documentation/docs/DIFFUSION_MODELS.md), [LANGUAGE_MODELS.md](../documentation/docs/LANGUAGE_MODELS.md), [AGENT_SYSTEMS.md](../documentation/docs/AGENT_SYSTEMS.md). Для первого доменного сценария сразу проверить **«3–5 строк кода»** (фабрика гиперграфа или хелпер, разумные умолчания).

### Часть 2: Стадия, мир, вселенная (после стабилизации Части 1)

| Фаза | Содержание | Детали |
|------|------------|--------|
| **7** | Стадия (state, маппинги, can_run) | [PHASE_7_STAGE.md](PHASE_7_STAGE.md) |
| **8** | Мир (цикл автор→среда→творец, state_schema) | [PHASE_8_WORLD.md](PHASE_8_WORLD.md) |
| **9** | Вселенная (миры, эфир, payload_spec) | [PHASE_9_UNIVERSE.md](PHASE_9_UNIVERSE.md) |
| **10** | Сериализация воркфлоу/стадии/мира/вселенной | [PHASE_10_SERIALIZATION_POLISH.md](PHASE_10_SERIALIZATION_POLISH.md) |

---

## В работе

- Старт с **фазы 0** по [TODO_PART1.md](TODO_PART1.md) и [PHASE_0_STRUCTURE.md](PHASE_0_STRUCTURE.md).

---

## Очередь

1. **Часть 1:** фазы 0 → 1 → 2 → 3 → 4 → 5 → 6 по порядку; после каждой фазы — приёмка по чек-листу в TODO_PART1 и при необходимости запись в DEV_LOG.
2. Домены и проверка «3–5 строк» для первого сценария (диффузия / LLM / агент — на выбор).
3. **Часть 2:** фазы 7–10 по [TODO_PART2.md](TODO_PART2.md) и [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) после стабилизации движка и хотя бы одного домена.

---

## Приоритеты (кратко)

1. **Фаза 0** — структура (yggdrasill/, tests/, pyproject.toml, pytest).
2. **Фаза 1** — фундамент (Block, Node, Port, реестр); без него ничего не запустить.
3. **Фаза 2** — движок (планировщик, буферы, исполнитель); первый run по графу.
4. **Фаза 3** — гиперграф задачи и **первый полный run(hypergraph, inputs)**; циклы.
5. **Фаза 4** — семь ролей и заглушки; сбор графов из узлов-задач.
6. **Фаза 5** — сериализация (конфиг + чекпоинт).
7. **Фаза 6** — воркфлоу.
8. Домены (диффузия, язык, агенты) и проверка UX «3–5 строк».
9. Часть 2 (фазы 7–10) — по готовности.
