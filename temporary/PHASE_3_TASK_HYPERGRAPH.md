# Фаза 3. Гиперграф задачи — полный технический план

Детальный технический план по **третьей фазе**: гиперграф задачи как полноценная собираемая и исполняемая единица. Цель — расширить движок (фаза 2) **API уровня задачи**: добавление узлов по **block_type** и **config** через реестр, построение из конфига (**from_config**), экспорт структуры (**to_config**), идентификаторы и метаданные (graph_id, graph_kind, metadata), контракт **run(hypergraph, inputs, **options) → outputs** через движок, подготовка к сериализации (state_dict/load_state_dict). Документ опирается на канон (01, 02, 03, HYPERGRAPH_ENGINE, SERIALIZATION), план реализации и референс (outdated_1).

**Связь с планом:** [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — Фаза 3.

**Канон:** [documentation/docs/03_TASK_HYPERGRAPH.md](../documentation/docs/03_TASK_HYPERGRAPH.md), [documentation/docs/01_FOUNDATION.md](../documentation/docs/01_FOUNDATION.md), [documentation/docs/02_ABSTRACT_TASK_NODES.md](../documentation/docs/02_ABSTRACT_TASK_NODES.md), [documentation/docs/HYPERGRAPH_ENGINE.md](../documentation/docs/HYPERGRAPH_ENGINE.md), [documentation/docs/SERIALIZATION.md](../documentation/docs/SERIALIZATION.md).

**Референс:** reference/outdated_1/foundation/graph.py (Graph, from_config, to_config, add_node(block_type, config), state_dict, load_state_dict).

**Язык:** русский.

---

## 1. Цель фазы 3

Реализовать **гиперграф задачи** как единицу, которую можно:

- **Собирать** из узлов-задач: добавлять узлы по **block_type** и **config** через реестр типов блоков (без ручного создания блоков и обёрток в коде).
- **Строить из конфига:** **from_config(config, registry?, validate?)** — полное восстановление структуры (nodes, edges, exposed_inputs, exposed_outputs, graph_id, graph_kind, metadata) через реестр.
- **Экспортировать структуру:** **to_config()** — словарь, пригодный для сериализации и повторного from_config (подготовка к фазе 5).
- **Выполнять** через движок: **run(hypergraph, inputs, **options) → outputs** (делегирование в engine.run); опции num_loop_steps, training, device, dry_run, callbacks и т.д.
- **Использовать как узел воркфлоу:** get_input_spec(include_dtype?), get_output_spec(include_dtype?), to(device), trainable_parameters(), set_trainable() — контракт для уровня воркфлоу и стадии (03 §8, 04).

**Результат фазы 3:** гиперграф задачи собирается из конфига или программно (add_node(block_type, config)), прогоняется через движок (run), экспортируется в конфиг; воспроизводимость задаётся конфигом + чекпоинтом (загрузка чекпоинта — фаза 5). Опционально: add_hyperedge(ends), infer_exposed_ports(), автосвязывание (auto_connect) — по объёму фазы.

---

## 2. Зависимости

- **Фаза 0:** структура репозитория, pytest. См. [PHASE_0_STRUCTURE.md](PHASE_0_STRUCTURE.md).
- **Фаза 1:** Port, AbstractBaseBlock, AbstractGraphNode, BlockRegistry; блоки создаются через **registry.build(config)** (config содержит block_type и параметры). См. [PHASE_1_FOUNDATION.md](PHASE_1_FOUNDATION.md).
- **Фаза 2:** Edge, Hypergraph (узлы, рёбра, exposed_inputs/outputs, execution_version), Validator, Planner, Executor, **run(hypergraph, inputs, **options) → outputs**. Гиперграф уже умеет add_node(node_id, node), add_edge(edge), expose_input/output, get_*; движок выполняет план. См. [PHASE_2_ENGINE.md](PHASE_2_ENGINE.md).

Фаза 3 **расширяет** класс Hypergraph методами уровня задачи: add_node(node_id, block_type, config, ...), from_config, to_config, run(), to(device), trainable_parameters(), set_trainable(), state_dict(), load_state_dict().

---

## 3. Что входит в фазу 3 и что нет

| Входит в фазу 3 | Не входит (фаза 4, 5 и далее) |
|-----------------|-------------------------------|
| add_node(node_id, block_type, config, registry?, pretrained?, trainable?, block_id?, **kwargs) — сборка узла-задачи через реестр (один объект Block+Node), добавление в Hypergraph. | Семь ролей узлов-задач как отдельные классы и регистрация backbone/..., inner_module/... (фаза 4). |
| add_node(node) — добавление готового узла (уже есть в фазе 2). | Agent_loop в исполнителе (можно заглушка или фаза 4). |
| from_config(config, registry?, validate?) — построение гиперграфа из словаря конфига. | Сериализация в файлы (save_config, save_checkpoint, load, load_from_checkpoint) — фаза 5. |
| to_config() — экспорт структуры в словарь (nodes, edges, exposed_*, graph_id, graph_kind, metadata, schema_version). | Полная дедупликация по block_id (один экземпляр на block_id при from_config) — можно заложить в фазе 3 или фаза 5. |
| graph_id, graph_kind, metadata (в т.ч. num_loop_steps) на гиперграфе. | |
| get_input_spec(include_dtype?), get_output_spec(include_dtype?) — для воркфлоу и пайплайнов. | |
| infer_exposed_ports() — вывод внешних входов/выходов по отсутствующим входящим/исходящим рёбрам. | |
| run(inputs, **options) — метод гиперграфа, делегирующий в engine.run(self, inputs, **options). | |
| to(device), trainable_parameters(), set_trainable(node_id, trainable). | |
| state_dict(), load_state_dict(state) — агрегат state_dict блоков по node_id (подготовка к фазе 5). | |
| Опционально: add_hyperedge(ends), автосвязывание (auto_connect) — заглушка или базовая логика. | |

---

## 4. Размещение кода и архитектура

Один класс **Hypergraph** в `yggdrasill/engine/structure.py`, расширенный методами фазы 3. Фаза 2 уже дала add_node(node_id, node), add_edge, expose_*, get_*; в фазе 3 добавляем в тот же класс:

- add_node(node_id, block_type, config, registry=None, pretrained=None, trainable=True, block_id=None, **kwargs) → node_id  
- from_config(config, registry=None, validate=False) → Hypergraph (classmethod)  
- to_config() → Dict  
- run(inputs, **options) → Dict  
- to(device), trainable_parameters(), set_trainable(node_id, trainable)  
- get_input_spec(include_dtype=False), get_output_spec(include_dtype=False)  
- infer_exposed_ports()  
- state_dict(), load_state_dict(state)

В документе предполагается **один класс Hypergraph** с двумя способами добавления узла: add_node(node_id, node) и add_node(node_id, block_type, config, ...).

**Файлы для изменений/добавлений:**

- `yggdrasill/engine/structure.py` — расширение Hypergraph: add_node с block_type/config, from_config, to_config, run, to, trainable_parameters, set_trainable, get_input_spec(include_dtype), get_output_spec(include_dtype), infer_exposed_ports, state_dict, load_state_dict.
- `yggdrasill/engine/__init__.py` — экспорт без изменений (Hypergraph уже экспортируется).
- Опционально: `yggdrasill/engine/config_loader.py` — _resolve_config_ref(config) для подгрузки конфига узла из файла (config["ref"] = "path/to.yaml").

**Тесты:**

- `tests/engine/test_task_hypergraph.py` (или расширить tests/engine/test_structure.py): from_config, to_config roundtrip, add_node(block_type, config), run после from_config, get_input_spec(include_dtype), infer_exposed_ports, state_dict/load_state_dict.

---

## 5. Добавление узла по block_type и config

### 5.1 Сигнатура

```python
def add_node(
    self,
    node_id: str,
    block_type: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    block_id: Optional[str] = None,
    pretrained: Optional[Union[str, Dict[str, Any]]] = None,
    trainable: bool = True,
    registry: Optional[BlockRegistry] = None,
    **kwargs: Any,
) -> str:
    """
    Добавить узел, создав блок через реестр по block_type и config.
    Возвращает node_id.
    """
```

**Сосуществование с фазой 2:** низкоуровневый вызов **add_node(node_id, node)** (добавление готового узла-задачи — объекта Block+Node) остаётся; при вызове **add_node(node_id, block_type, config=..., ...)** внутри вызывается реестр, который возвращает один объект узла-задачи (Block+Node), и вызывается add_node(node_id, node). Реализация может быть одной перегруженной функцией (по типу второго аргумента: узел vs str) или отдельным методом add_node_from_config; в API для пользователя достаточно одного add_node с двумя вариантами вызова.

**Параметры:**

- **node_id** — уникальный в рамках гиперграфа идентификатор узла (строка, непустая после strip).
- **block_type** — тип блока для реестра (например "backbone/identity", "converter/vae"). Обязателен при данном варианте вызова.
- **config** — словарь конфигурации блока (передаётся в registry.build вместе с block_type). Может содержать block_id, параметры архитектуры и т.д. Если None — пустой dict.
- **block_id** — опциональный идентификатор блока (для дедупликации и чекпоинтов). Если не передан, блок может сгенерировать свой (см. Phase 1). При повторном использовании одного блока в нескольких узлах задаётся один block_id.
- **pretrained** — опционально: путь к файлу чекпоинта (строка) или готовый state_dict (dict). После создания блока вызывается block.load_state_dict(pretrained) или загрузка из файла по соглашению (например JSON). В фазе 3 достаточно поддержать dict; загрузка из файла может быть в фазе 5.
- **trainable** — флаг обучаемости узла (по умолчанию True). Сохраняется в гиперграфе и учитывается в trainable_parameters().
- **registry** — реестр типов блоков. Если None — BlockRegistry.global_registry().
- **kwargs** — дополнительные аргументы (например auto_connect для фазы 4), пока игнорируются или сохраняются.

### 5.2 Алгоритм

1. Нормализовать node_id (strip); проверить непустоту. Проверить, что node_id не занят (иначе ValueError).
2. Собрать build_config: {"block_type": block_type, "node_id": node_id, "block_id": block_id, **(config or {})}. Убрать из config ключи, не относящиеся к узлу-задаче, если нужно.
3. registry = registry or BlockRegistry.global_registry(); **node = registry.build(build_config)**. Реестр для типов узлов-задач (фаза 4) возвращает один объект Block+Node; фабрика принимает node_id, block_id, config и создаёт экземпляр узла-задачи. Отдельной обёртки AbstractGraphNode(block=...) не используется.
4. При необходимости установить node_id на объекте (если не передан в конструктор). Вызвать существующий add_node(node_id, node) (низкоуровневый метод фазы 2).
5. Записать trainable в внутренний словарь _node_trainable[node_id] = trainable (если такой словарь ведётся; иначе атрибут на узле или общая политика).
6. Если pretrained задан: если dict — node.load_state_dict(pretrained, strict=False); если str — по соглашению загрузить из файла (фаза 5 или простой json.load).
7. Вернуть node_id.

### 5.3 Ошибки

- node_id пустой или уже существует — ValueError.
- block_type не зарегистрирован — исключение из registry.build (KeyError или аналог).
- config некорректный для блока — исключение из registry.build.

### 5.4 Тесты

- Успешное добавление: add_node("A", "test/identity", config={}) с зарегистрированным типом test/identity (узлы-задачи, фаза 4); get_node("A") возвращает узёл-задачу (объект Block+Node).
- С pretrained (dict): после add_node с pretrained=state_dict блок возвращает ожидаемые выходы.
- Дубликат node_id — ValueError.
- Неизвестный block_type — ожидаемое исключение от реестра.

---

## 6. Построение из конфига (from_config)

### 6.1 Формат конфига

Конфиг — словарь со следующими ключами (канон 03 §9.1, SERIALIZATION):

- **nodes** — список словарей. Каждый элемент: **node_id** (str), **block_type** (str), опционально **block_id**, **config** (dict или ref, см. ниже), **trainable** (bool, по умолчанию True).
- **edges** — список словарей: **source_node**, **source_port**, **target_node**, **target_port** (все строки).
- **exposed_inputs** — список словарей: **node_id**, **port_name**, опционально **name**.
- **exposed_outputs** — список словарей: **node_id**, **port_name**, опционально **name**.
- **graph_id** — строка (опционально, по умолчанию "graph").
- **graph_kind** — строка (опционально): "diffusion", "agent", "rag", "generic" и т.д.
- **metadata** — словарь (опционально): num_loop_steps, описание и т.д.
- **schema_version** — строка (опционально): для миграций и проверки совместимости.

Конфиг узла в **nodes[].config** может быть подставлен из файла: если config = {"ref": "path/to/file.yaml"} (или .json), загрузить содержимое файла и использовать как config узла (референс: _resolve_config_ref). Иначе config передаётся в registry.build как есть.

### 6.2 Сигнатура

```python
@classmethod
def from_config(
    cls,
    config: Dict[str, Any],
    *,
    registry: Optional[BlockRegistry] = None,
    validate: bool = False,
) -> "Hypergraph":
    """
    Построить гиперграф из словаря конфига.
    Если validate=True, после сборки вызвать validate(); при ошибках — исключение.
    """
```

### 6.3 Алгоритм

1. registry = registry or BlockRegistry.global_registry().
2. Создать экземпляр: g = cls(graph_id=config.get("graph_id", "graph")).
3. Установить graph_kind и metadata: g.graph_kind = config.get("graph_kind"); g.metadata = dict(config.get("metadata", {})).
4. **Узлы:** для каждой записи nc в config.get("nodes", []): node_id = nc["node_id"]; block_type = nc["block_type"]; node_cfg = nc.get("config") или {}; если node_cfg — dict с единственным ключом "ref", подставить содержимое файла (_resolve_config_ref); build_config = {"block_type": block_type, "node_id": node_id, "block_id": nc.get("block_id"), **node_cfg}; **node = registry.build(build_config)** (реестр возвращает узёл-задачу — объект Block+Node; фаза 4 регистрирует типы backbone/identity и т.д.); g.add_node(node_id, node); g._node_trainable[node_id] = nc.get("trainable", True).
5. **Рёбра:** для каждой записи ec в config.get("edges", []): g.add_edge(Edge(source_node=ec["source_node"], source_port=ec["source_port"], target_node=ec["target_node"], target_port=ec["target_port"])).
6. **Exposed:** для каждой записи в config.get("exposed_inputs", []): g.expose_input(entry["node_id"], entry["port_name"], entry.get("name")). Аналогично exposed_outputs.
7. Если validate: result = validate(g); если result.errors — выбросить исключение (например ValueError с текстом ошибок) или вернуть результат по политике.
8. Вернуть g.

### 6.4 Разрешение ссылок в конфиге узла (_resolve_config_ref)

Если config узла — словарь вида {"ref": "path/to/file.yaml"} (единственный ключ "ref"), загрузить файл и вернуть его содержимое (YAML или JSON). Иначе вернуть config как есть. Это позволяет выносить большие конфиги блоков в отдельные файлы (канон 03 §4.4, референс graph._resolve_config_ref).

### 6.5 Тесты

- Минимальный конфиг: nodes (один узел), edges (пусто), exposed_inputs/outputs (один порт). from_config даёт гиперграф с одним узлом; run с одним входом возвращает ожидаемый выход.
- Полный конфиг: несколько узлов, рёбра, exposed. from_config → to_config → from_config: структура совпадает (roundtrip).
- validate=True и невалидный конфиг (например порт отсутствует) — исключение.
- config узла с "ref": загружается внешний файл (мок или временный файл в тесте).

---

## 7. Экспорт структуры (to_config)

### 7.1 Назначение

Вернуть словарь, полностью описывающий структуру гиперграфа (без весов): nodes, edges, exposed_inputs, exposed_outputs, graph_id, graph_kind, metadata, schema_version. Используется для сериализации в файл (фаза 5), отладки и roundtrip from_config(to_config()) == эквивалентная структура.

### 7.2 Сигнатура

```python
def to_config(self) -> Dict[str, Any]:
    """Сериализация структуры (без весов)."""
```

### 7.3 Содержимое

- **schema_version** — константа (например "1.0") для совместимости и миграций.
- **graph_id** — self.graph_id.
- **nodes** — список dict: для каждого node_id, node (узел-задача = Block+Node): {"node_id": node_id, "block_type": node.block_type, "block_id": node.block_id, "config": node.config, "trainable": self._node_trainable.get(node_id, True)}.
- **edges** — список dict: {"source_node", "source_port", "target_node", "target_port"} для каждого Edge.
- **exposed_inputs** — список dict: {"node_id", "port_name", "name"?} (копия _exposed_inputs).
- **exposed_outputs** — список dict: {"node_id", "port_name", "name"?}.
- **graph_kind** — если задан, добавить ключ graph_kind.
- **metadata** — если непустой, добавить metadata.

Блоки должны предоставлять **block_type**, **block_id**, **config** (сериализуемый dict). По канону 01 блок имеет block_type и block_id; config хранится в блоке (Phase 1).

### 7.4 Тесты

- to_config() после add_node(block_type, config) и add_edge: в конфиге присутствуют nodes с block_type и config, edges совпадают с добавленными.
- Roundtrip: g2 = Hypergraph.from_config(g1.to_config(), registry=...); совпадение node_ids, edges, exposed_inputs/outputs, graph_id, graph_kind, metadata (без проверки идентичности объектов блоков).

---

## 8. Идентификаторы и метаданные

- **graph_id** — строковый идентификатор гиперграфа (по умолчанию "graph"). Задаётся при создании Hypergraph(graph_id=...) или из config["graph_id"] в from_config. Используется при сериализации и в логах.
- **graph_kind** — опциональная подсказка типа задачи: "diffusion", "agent", "rag", "generic". Планировщик и валидатор могут использовать для шаблонов (03 §4.4). Устанавливается через setter или из config["graph_kind"].
- **metadata** — словарь произвольных метаданных. Обязательно поддерживать **num_loop_steps** (int) как значение по умолчанию для числа итераций цикла при run (если не передано в options). Остальное: описание, версия, параметры run — по соглашению.

Хранение: в Hypergraph атрибуты _graph_id, _graph_kind, _metadata (или property metadata как в Phase 2). При to_config все попадают в выходной словарь; при from_config восстанавливаются.

---

## 9. Контракт run и интеграция с движком

Гиперграф задачи выполняется тем же движком, что и в фазе 2. Вызов: **run(hypergraph, inputs, **options) → outputs**.

### 9.1 Метод run на гиперграфе

Удобно предоставить метод экземпляра, делегирующий в движок:

```python
def run(
    self,
    inputs: Dict[str, Any],
    *,
    training: bool = False,
    num_loop_steps: Optional[int] = None,
    device: Optional[Any] = None,
    callbacks: Optional[List[Callable[..., None]]] = None,
    dry_run: bool = False,
    validate_before: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Выполнить гиперграф; делегирует в engine.run(self, inputs, ...)."""
    from yggdrasill.engine.executor import run as _run
    return _run(
        self, inputs,
        training=training, num_loop_steps=num_loop_steps, device=device,
        callbacks=callbacks, dry_run=dry_run, validate_before=validate_before,
        **kwargs,
    )
```

**num_loop_steps:** если не передан в options, движок берёт self.metadata.get("num_loop_steps", 1) (канон 03 §8.2, Phase 2 executor).

### 9.2 Входы и выходы

- **inputs** — словарь: ключи = name (если задан в expose_input) или "node_id:port_name"; значения = данные для соответствующих портов.
- **outputs** — словарь с ключами по get_output_spec(); значения — результаты с внешних выходов после выполнения плана.

Тесты: после from_config или add_node(block_type, config) вызвать hypergraph.run(inputs) и проверить наличие и значение ключей в outputs.

---

## 10. Спецификации входов/выходов и совместимость с воркфлоу

Воркфлоу (уровень 0.4) и стадия опираются на то, что узел (гиперграф задачи) предоставляет **get_input_spec** и **get_output_spec** для согласования портов и типов (04 §3.4, контракт для stage).

### 10.1 get_input_spec(include_dtype=False)

Возвращает список записей для внешних входов гиперграфа. Каждая запись — dict: **node_id**, **port_name**, опционально **name**. Если **include_dtype=True**, добавить в каждую запись поле **dtype** (имя типа порта из Port.dtype, например "TENSOR", "TEXT") для проверки совместимости на уровне воркфлоу. Реализация: обойти _exposed_inputs; для include_dtype взять у узла блок, у блока порт по port_name (из get_input_ports()), записать dtype.

### 10.2 get_output_spec(include_dtype=False)

Аналогично для внешних выходов: список dict с node_id, port_name, name?, при include_dtype — dtype из выходного порта.

Сигнатуры (референс graph.py):

```python
def get_input_spec(self, include_dtype: bool = False) -> List[Dict[str, Any]]: ...
def get_output_spec(self, include_dtype: bool = False) -> List[Dict[str, Any]]: ...
```

Тесты: get_input_spec(include_dtype=True) содержит ключ "dtype" в каждой записи при наличии портов с dtype.

---

## 11. Вывод внешних портов (infer_exposed_ports)

**infer_exposed_ports()** — установить _exposed_inputs и _exposed_outputs по правилу: внешний вход = входной порт узла, на который **нет входящего ребра** от другого узла; внешний выход = выходной порт, от которого **нет исходящего ребра** к другому узлу (канон 03 §4.3, референс graph.infer_exposed_ports). Вызов перезаписывает текущие списки exposed. Используется, когда граф собран по рёбрам и не заданы явные expose_input/expose_output.

Алгоритм:

- _exposed_inputs: для каждого узла и каждого его входного порта (get_input_ports()) проверить, есть ли входящее ребро (get_edges_in(node_id) с target_port == port.name). Если нет — добавить запись (node_id, port.name, name=None).
- _exposed_outputs: для каждого узла и каждого выходного порта проверить, есть ли исходящее ребро. Если нет — добавить запись.

Тесты: граф A → B → C без явного expose; после infer_exposed_ports() входы — входные порты A, выходы — выходные порты C; get_input_spec/get_output_spec не пусты.

---

## 12. Устройство, обучаемость, state_dict

### 12.1 to(device)

Переместить все блоки гиперграфа на устройство (CPU/GPU), если у блока есть метод .to(device). Возвращать self для цепочки вызовов. Используется перед run при необходимости (03 §8.2, 04).

```python
def to(self, device: Any) -> "Hypergraph":
    for nid in self.node_ids:
        node = self.get_node(nid)
        if node is not None and hasattr(node, "to") and callable(getattr(node, "to")):
            node.to(device)
    return self
```

### 12.2 trainable_parameters(), set_trainable(node_id, trainable)

- **trainable_parameters()** — итератор по параметрам (например для оптимизатора): обойти узлы, для которых _node_trainable.get(node_id, True) == True, и отдавать node.trainable_parameters() (узел-задача имеет интерфейс блока; канон 03 §12.5, DOMAINS_DEPLOYMENT_TRAINING). Общие блоки (один block_id в нескольких узлах) не дублировать параметры — при реализации дедупликации по block_id учитывать один раз на block_id.
- **set_trainable(node_id, trainable: bool)** — установить флаг обучаемости узла. При node_id не из графа — ValueError.

### 12.3 state_dict(), load_state_dict(state)

- **state_dict()** — агрегат state_dict всех узлов-задач (узел = Block+Node, у него есть state_dict от блока). Ключ — node_id (или block_id при дедупликации, канон SERIALIZATION §4.4). Простая реализация: {node_id: node.state_dict() for node_id, node in self._nodes.items() if hasattr(node, 'state_dict') and node.state_dict()}. При общем блоке (несколько узлов с одним block_id) — одна запись по block_id (фаза 5 уточняет).
- **load_state_dict(state: Dict, strict: bool = True)** — для каждого node_id в self._nodes загрузить state[node_id] в node.load_state_dict(...). Если strict, проверить, что все ключи state соответствуют node_id в графе.

Подготовка к фазе 5: сохранение/загрузка чекпоинта в файлы будет использовать state_dict и load_state_dict.

Тесты: после run изменить состояние блока через load_state_dict; следующий run даёт другой результат (если блок состояние использует).

---

## 13. Гиперрёбра арности > 2 (опционально)

Канон 03 §4.2: **add_hyperedge(ends)** — гиперребро произвольной арности; ends — список пар (node_id, port_name) с разметкой источник/приёмник. В фазе 2 движок оперирует рёбрами Edge (арность 2). Варианты реализации в фазе 3:

- **Вариант 1:** add_hyperedge(ends) раскладывать в несколько Edge: один источник → несколько приёмников = несколько рёбер (источник, порт) → (target_i, port_i); несколько источников → один приёмник = несколько рёбер, агрегация на приёмнике по политике порта (уже в движке при нескольких входящих рёбрах).
- **Вариант 2:** ввести тип HyperEdge (список концов с ролью source/target), хранить в Hypergraph отдельно; планировщик и буферы движка расширить для чтения HyperEdge (запись в буферы по всем приёмникам от одного источника; сбор из нескольких источников с агрегацией). Для «полного» соответствия канону — вариант 2; для минимальной фазы 3 — вариант 1.

Документировать в API: add_hyperedge(ends: List[Tuple[str, str, str]], role: "source" | "target") или ends как список dict с полями node_id, port_name, role. При реализации варианта 1 внутри создавать Edge для каждой пары (источник, приёмник).

---

## 14. Автосвязывание (auto_connect) — опционально

По канону 03 §4.1 при добавлении узла может использоваться **автосвязывание**: по типу узла-задачи (backbone, inner_module, conjector, ...) система выводит типичные связи с уже существующими узлами и создаёт гиперрёбра. Это требует знаний о ролях (02_ABSTRACT_TASK_NODES); фаза 4 вводит семь ролей. В фазе 3 можно:

- Добавить параметр **auto_connect: bool = False** в add_node(node_id, block_type, config, ...); при True вызывать хук или заглушку (например пустая функция или попытка импорта из yggdrasill.task_nodes с use_task_node_auto_connect, как в референсе). Без фазы 4 автосвязывание не создаёт рёбер; с фазой 4 — подключается реальная логика.
- Либо не добавлять auto_connect в фазу 3 и ввести в фазе 4.

Тесты: при auto_connect=True и отсутствии модуля task_nodes — add_node не падает; рёбра создаются только явно или через from_config.

---

## 15. Порядок реализации и тесты

### 15.1 Порядок реализации

1. В Hypergraph добавить _node_trainable: Dict[str, bool], graph_id (уже может быть), graph_kind, metadata (property/setter при необходимости).
2. Реализовать **add_node(node_id, block_type, config, ...)** с вызовом registry.build и низкоуровневого add_node(node_id, node); сохранять trainable; опционально pretrained.
3. Реализовать **to_config()** и **from_config()**; _resolve_config_ref для config["ref"].
4. Реализовать **get_input_spec(include_dtype)**, **get_output_spec(include_dtype)** (расширить существующие, если уже возвращают только список без dtype).
5. Реализовать **infer_exposed_ports()**.
6. Добавить метод **run(self, inputs, **options)** с делегированием в engine.run(self, inputs, **options).
7. Реализовать **to(device)**, **trainable_parameters()**, **set_trainable(node_id, trainable)**.
8. Реализовать **state_dict()**, **load_state_dict(state)**.
9. Опционально: add_hyperedge(ends), параметр auto_connect в add_node.

### 15.2 Тесты (сводка)

| Сценарий | Ожидание |
|----------|----------|
| add_node("A", "test/identity", config={}) с зарегистрированным типом | Узел A в графе, get_node("A").block_type == "test/identity" (узел-задача = Block+Node). |
| from_config(minimal_config) | Гиперграф с узлами и рёбрами из конфига; run(inputs) возвращает outputs по exposed_outputs. |
| to_config() после сборки | Словарь с nodes, edges, exposed_inputs, exposed_outputs, graph_id, schema_version. |
| from_config(g.to_config()) roundtrip | Совпадение структуры (node_ids, edges, exposed), run даёт тот же результат при тех же весах. |
| get_input_spec(include_dtype=True) | Записи содержат ключ "dtype" при наличии портов. |
| infer_exposed_ports() на графе A→B→C без expose | Входы = порты A без входящих рёбер, выходы = порты C без исходящих рёбер. |
| hypergraph.run(inputs) после from_config | Результат совпадает с вызовом engine.run(hypergraph, inputs). |
| state_dict / load_state_dict | state_dict возвращает dict по node_id; load_state_dict восстанавливает веса; следующий run отражает изменение. |
| Дубликат node_id в add_node | ValueError. |
| Неизвестный block_type в from_config | Исключение от registry.build. |

---

## 16. Приёмочные критерии и граничные случаи

### 16.1 Приёмочные критерии

- [ ] add_node(node_id, block_type, config, registry=...) создаёт узел с блоком из реестра; узел участвует в run.
- [ ] from_config(config, registry, validate=True) строит гиперграф; при невалидном конфиге при validate=True — исключение.
- [ ] to_config() возвращает словарь, пригодный для from_config; roundtrip сохраняет структуру.
- [ ] hypergraph.run(inputs, num_loop_steps=K) выполняется через движок; outputs соответствуют get_output_spec().
- [ ] get_input_spec(include_dtype=True) и get_output_spec(include_dtype=True) возвращают записи с dtype при наличии портов.
- [ ] infer_exposed_ports() выставляет exposed_inputs/exposed_outputs по отсутствующим рёбрам.
- [ ] to(device), trainable_parameters(), set_trainable работают; state_dict/load_state_dict агрегируют/загружают по node_id.
- [ ] Все тесты tests/engine для фазы 3 (from_config, to_config, add_node(block_type, config), run, spec, infer, state_dict) проходят.

### 16.2 Граничные случаи

| Случай | Ожидание |
|--------|----------|
| Пустой config["nodes"] | Гиперграф без узлов; from_config не падает; run с пустым планом возвращает пустой dict или по exposed_outputs (пусто). |
| config узла без block_type | from_config падает (обязательное поле). |
| config["ref"] на несуществующий файл | Ошибка при разрешении (или пустой config по соглашению). |
| Два узла с одинаковым block_id | В фазе 3 можно хранить два экземпляра блока; в фазе 5 — дедупликация, один экземпляр на block_id (SERIALIZATION §4.4). |
| run без явных exposed_inputs/exposed_outputs | Движок использует get_input_spec/get_output_spec; если списки пусты (и infer_exposed_ports не вызывался), run может вернуть пустой outputs или ошибку валидации. |
| pretrained с путём к несуществующему файлу | Ошибка загрузки в add_node или отложенная до run (по соглашению). |

---

## 17. Референс: outdated_1 (graph.py)

- **add_node(node_id, block_type, config, pretrained, registry, auto_connect, **kwargs)** — два аргумента (node_id, block_type) или один (Node/block/block_type). Build через registry; add_node(Node(...)); _node_trainable; pretrained загрузка; auto_connect через _ensure_auto_connect (task_nodes). Использовать как образец сигнатуры и порядка шагов.
- **from_config(config, registry, validate)** — создание Graph, установка graph_kind и metadata; цикл по nodes (node_cfg с _resolve_config_ref), registry.build, add_node(Node); цикл по edges add_edge(Edge); восстановление _exposed_inputs/_exposed_outputs; при validate — g.validate(strict=True).
- **to_config()** — schema_version, graph_id, nodes (node_id, block_type, block_id, config, trainable), edges, exposed_inputs, exposed_outputs, graph_kind, metadata.
- **_resolve_config_ref(config)** — если config == {"ref": path}, загрузить YAML/JSON из path и вернуть содержимое.
- **get_input_spec(include_dtype)**, **get_output_spec(include_dtype)** — при include_dtype добавить dtype из port в каждую запись.
- **infer_exposed_ports()** — обнулить _exposed_inputs/_exposed_outputs; для каждого узла входные порты без входящего ребра → exposed_inputs; выходные без исходящего → exposed_outputs.
- **run(inputs, ...)** — делегирование в executor.run(self, inputs, ...).
- **to(device)**, **trainable_parameters()**, **set_trainable(node_id, trainable)**.
- **state_dict()**, **load_state_dict(state)** — агрегат по node_id.
- **load_from_checkpoint**, **save_config**, **save_checkpoint**, **save**, **load** — фаза 5; в фазе 3 достаточно state_dict/load_state_dict и to_config/from_config.

Отличия от референса в нашей фазе 3: класс Hypergraph (не Graph); узел — AbstractGraphNode (Phase 1); реестр — BlockRegistry; движок в engine.run. Имена методов и контракт to_config/from_config сохраняем совместимыми с каноном и референсом.

---

## Итог

Фаза 3 задаёт **полный технический план гиперграфа задачи**: расширение Hypergraph методами уровня задачи (add_node по block_type и config, from_config, to_config), идентификаторы и метаданные (graph_id, graph_kind, metadata), контракт run через движок, get_input_spec/get_output_spec с опцией dtype, infer_exposed_ports, to(device), trainable_parameters, set_trainable, state_dict/load_state_dict. Реализация в указанном порядке с тестами (from_config, to_config roundtrip, run после сборки из конфига) обеспечивает собираемый и исполняемый гиперграф задачи; сериализация в файлы (save/load) — фаза 5. Документ опирается на канон (01, 02, 03, HYPERGRAPH_ENGINE, SERIALIZATION) и референс outdated_1 (graph.py).
