# Фаза 6. Воркфлоу — полный технический план

Детальный технический план по **шестой фазе**: воркфлоу как **гиперграф гиперграфов** — узлы воркфлоу суть **целые гиперграфы задач**, гиперрёбра воркфлоу соединяют **внешние выходы** одного гиперграфа с **внешними входами** другого. Цель — применить **тот же гиперграфовый движок** (валидатор, планировщик, буферы, исполнитель) к уровню воркфлоу без смены ядра: узел = гиперграф, вызов узла = run(hypergraph, inputs) → outputs. Документ опирается на канон (04_WORKFLOW) и фазы 2, 3, 5.

**Связь с планом:** [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — Фаза 6.

**Канон:** [documentation/docs/04_WORKFLOW.md](../documentation/docs/04_WORKFLOW.md), [documentation/docs/03_TASK_HYPERGRAPH.md](../documentation/docs/03_TASK_HYPERGRAPH.md), [documentation/docs/HYPERGRAPH_ENGINE.md](../documentation/docs/HYPERGRAPH_ENGINE.md), [documentation/docs/SERIALIZATION.md](../documentation/docs/SERIALIZATION.md).

**Язык:** русский.

---

## 1. Цель фазы 6

Реализовать **воркфлоу** так, чтобы:

- **Воркфлоу** — гиперграф, узлы которого — **гиперграфы задач** (экземпляры Hypergraph из фазы 3). Идентификатор узла на уровне воркфлоу — **graph_id** (в коде и конфиге используется то же имя, что и node_id в движке; для ясности в документе — graph_id).
- **Гиперрёбра воркфлоу** соединяют **внешний выход** одного гиперграфа с **внешним входом** другого (формат: source_graph_id, source_port, target_graph_id, target_port). Данные между задачами передаются по этим рёбрам.
- **Тот же движок** (Validator, Planner, Executor) применяется к воркфлоу: структура воркфлоу реализует тот же контракт, что и Hypergraph, с точки зрения движка (node_ids → get_node(id), get_edges(), get_input_spec(), get_output_spec()); разница только в том, что get_node(graph_id) возвращает **гиперграф**, у которого есть run(inputs) → outputs, get_input_spec(), get_output_spec().
- **Циклы между гиперграфами** поддерживаются: планировщик строит итеративную схему (начальная фаза → циклическая фаза K раз → конечная фаза); число K задаётся конфигом или опцией run (num_loop_steps, num_workflow_steps).
- **Построение:** add_node(graph_id, hypergraph), add_node(graph_id, config, registry), add_edge(...), expose_input/expose_output, from_config, to_config; **сериализация:** конфиг воркфлоу + чекпоинт как агрегат чекпоинтов гиперграфов (save/load по аналогии с фазой 5).

**Результат фазы 6:** можно собрать воркфлоу из нескольких гиперграфов задач, вызвать run(workflow, inputs) и получить outputs; сохранить и загрузить воркфлоу (конфиг + чекпоинты гиперграфов); воспроизводимость при тех же конфиге, чекпоинтах и входах.

---

## 2. Зависимости

- **Фаза 0:** структура репозитория, pytest. См. [PHASE_0_STRUCTURE.md](PHASE_0_STRUCTURE.md).
- **Фаза 1:** Port, AbstractBaseBlock, AbstractGraphNode, BlockRegistry. См. [PHASE_1_FOUNDATION.md](PHASE_1_FOUNDATION.md).
- **Фаза 2:** Edge, Hypergraph (базовое хранилище), Validator, Planner, Executor, run(hypergraph, inputs, **options). Движок оперирует структурой с node_ids, get_node(id), get_edges(), get_input_spec(), get_output_spec(); узел имеет run(inputs) → outputs. См. [PHASE_2_ENGINE.md](PHASE_2_ENGINE.md).
- **Фаза 3:** Hypergraph с add_node(block_type, config), from_config, to_config, run(), get_input_spec(), get_output_spec(), state_dict(), load_state_dict(). Гиперграф задачи сам реализует контракт «узла» для движка (run, get_input_spec, get_output_spec). См. [PHASE_3_TASK_HYPERGRAPH.md](PHASE_3_TASK_HYPERGRAPH.md).
- **Фаза 5:** Сериализация гиперграфа (save, load, save_config, save_checkpoint, load_from_checkpoint). См. [PHASE_5_SERIALIZATION.md](PHASE_5_SERIALIZATION.md).

Фаза 6 вводит класс **Workflow** и обеспечивает, что движок может выполнять воркфлоу так же, как гиперграф задачи: под «узлом» подразумевается гиперграф, вызов — run(hypergraph, inputs).

---

## 3. Что входит в фазу 6 и что нет

| Входит в фазу 6 | Не входит (другие фазы) |
|-----------------|--------------------------|
| Класс **Workflow**: хранилище graph_id → Hypergraph, рёбра между внешними портами гиперграфов, exposed_inputs/exposed_outputs воркфлоу, execution_version. | Стадия (фаза 7): state, state_input_map, state_output_map, can_run — уровень над воркфлоу. |
| add_node(graph_id, hypergraph), add_node(graph_id, config, registry?) — добавление гиперграфа как узла (из экземпляра или из конфига/ref). | Изменение ядра движка (Validator, Planner, Executor) — движок остаётся общим; воркфлоу реализует тот же интерфейс, что и Hypergraph для движка. |
| add_edge(source_graph_id, source_port, target_graph_id, target_port), remove_edge, remove_node. | Сериализация стадии, мира, вселенной. |
| expose_input(graph_id, port_name, name?), expose_output(graph_id, port_name, name?). | |
| get_input_spec(include_dtype?), get_output_spec(include_dtype?) — по внешним портам гиперграфов. | |
| **Контракт для движка:** workflow.node_ids (graph_ids), get_node(graph_id) → Hypergraph, get_edges() → List[Edge], get_input_spec(), get_output_spec(), metadata (num_loop_steps и т.д.). Движок вызывает get_node(id).run(inputs); Hypergraph.run уже есть. | |
| from_config(config, registry?, validate?), to_config() — конфиг воркфлоу (graphs, edges, exposed_*, workflow_id, workflow_kind, metadata, schema_version). | |
| run(workflow, inputs, **options) — делегирование в engine.run(workflow, inputs, ...); тот же executor, что и для Hypergraph. | |
| Поддержка **циклов между гиперграфами**: планировщик строит итеративную схему (как для гиперграфа задачи с циклом); num_loop_steps / num_workflow_steps из metadata или options. | |
| state_dict(), load_state_dict(state) — агрегат по graph_id → state_dict гиперграфа; to(device), trainable_parameters(), set_trainable(graph_id, bool). | |
| **Сериализация воркфлоу:** save(path), save_config(path), save_checkpoint(path), load(path, registry?, ...), load_config(path, ...), load_from_checkpoint(path). Конфиг воркфлоу + чекпоинты всех гиперграфов (по graph_id); при общем ref — один экземпляр, один чекпоинт. | |
| infer_exposed_ports() — опционально: вывод внешних входов/выходов воркфлоу по отсутствию входящих/исходящих рёбер. | |
| Тесты: roundtrip to_config/from_config, run цепочки из 2–3 гиперграфов, цикл между гиперграфами (K шагов), save/load воркфлоу, дедупликация при общем ref. | |

---

## 4. Размещение кода и архитектура

**Вариант архитектуры:** воркфлоу использует **тот же движок** (engine), что и гиперграф задачи. Для этого движок должен оперировать **абстракцией структуры**, а не только классом Hypergraph: любой объект с методами node_ids, get_node(id), get_edges(), get_input_spec(), get_output_spec(), metadata (и при необходимости get_edges_in(id), get_edges_out(id)) может быть передан в run(). Тогда run(hypergraph, inputs) и run(workflow, inputs) вызывают одну и ту же функцию run(structure, inputs), где structure — Hypergraph или Workflow.

**Файлы:**

- **yggdrasill/workflow/** — новый пакет фазы 6.
  - **__init__.py** — экспорт Workflow, при необходимости WorkflowEdge (или используется engine.Edge с полями source_node/source_port/target_node/target_port; на уровне воркфлоу source_node = graph_id).
  - **structure.py** — класс Workflow: _nodes (graph_id → Hypergraph), _edges, _exposed_inputs, _exposed_outputs, _execution_version, workflow_id, workflow_kind, metadata; add_node, add_edge, expose_*, get_node, get_edges, get_input_spec, get_output_spec, get_edges_in/out, node_ids; from_config, to_config; run (делегирует в engine.run(self, ...)); state_dict, load_state_dict; to(device), trainable_parameters(), set_trainable(graph_id, bool).
  - **io.py** (или в structure.py) — save, save_config, save_checkpoint, load, load_config, load_from_checkpoint для воркфлоу (по аналогии с фазой 5).

**Использование Edge:** на уровне воркфлоу ребро соединяет (graph_id, port_name) с (graph_id, port_name). Тип Edge из engine (source_node, source_port, target_node, target_port) подходит без изменений: source_node = source_graph_id, target_node = target_graph_id. Внутри Workflow храним Edge из engine; при валидации проверяем, что source_port — внешний выход гиперграфа source_graph_id, target_port — внешний вход гиперграфа target_graph_id.

**Исполнитель:** в фазе 2 executor.run(hypergraph, inputs) вызывает get_node(node_id).run(inputs). Для воркфлоу get_node(graph_id) возвращает Hypergraph; hypergraph.run(inputs) уже реализован в фазе 3. Поэтому executor не нужно менять — достаточно передать workflow в run(workflow, inputs). Условие: у Workflow те же методы, что нужны движку (node_ids, get_node, get_edges, get_input_spec, get_output_spec, get_edges_in, get_edges_out, metadata). Список имён узлов: в Workflow это graph_ids; свойство node_ids для совместимости с движком может возвращать set(_nodes.keys()).

**Валидатор и планировщик:** они принимают структуру с get_node, get_edges, get_input_spec, get_output_spec. Для воркфлоу валидатор должен проверять, что (graph_id, port_name) в рёбрах и в exposed_* ссылаются на **внешние** порты соответствующего гиперграфа (get_input_spec/get_output_spec гиперграфа), а не на внутренние порты узлов. Реализация: при валидации воркфлоу для каждого graph_id получать гиперграф через get_node(graph_id) и проверять port_name по hypergraph.get_input_spec() / get_output_spec(). Либо вынести общую логику валидатора в функцию, принимающую «структуру» и функцию получения портов узла (для Hypergraph — порты узла-задачи, для Workflow — внешние порты гиперграфа). В фазе 6 допустимо реализовать валидатор воркфлоу отдельно (WorkflowValidator), проверяющий графы и порты по get_node(graph_id).get_input_spec() и get_output_spec(), и вызывать его из run(workflow, ..., validate_before=True) вместо общего validate(workflow), если общий валидатор заточен под узлы-задачи. Предпочтительно: **обобщить валидатор** так, чтобы он принимал структуру и для каждого node_id (или graph_id) вызывал get_node(id) и у полученного узла запрашивал get_input_ports()/get_output_ports() или get_input_spec()/get_output_spec() — тогда один валидатор подходит и для Hypergraph, и для Workflow. В Phase 2 валидатор, возможно, опирается на get_node(node_id) и порты узла-задачи; для Workflow get_node возвращает Hypergraph, у которого нет get_input_ports(), но есть get_input_spec()/get_output_spec(). Поэтому в спецификации фазы 6: либо движок (валидатор) уже использует get_input_spec/get_output_spec структуры и порты узла получает через get_node(id).get_input_spec() и get_node(id).get_output_spec(), либо воркфлоу при валидации использует отдельную проверку. Документируем оба варианта и рекомендуем единый контракт: **узел (гиперграф или задача) предоставляет get_input_spec() и get_output_spec()** для списка внешних портов; валидатор для каждого id вызывает structure.get_node(id) и spec = node.get_input_spec() / node.get_output_spec() и проверяет port_name по spec. Тогда и Hypergraph (фаза 3), и Workflow в фазе 6 удовлетворяют этому контракту.

---

## 5. Класс Workflow — хранилище структуры

### 5.1 Внутреннее состояние

- **_nodes: Dict[str, Hypergraph]** — graph_id → гиперграф задачи (экземпляр Hypergraph из yggdrasill.engine или yggdrasill.hypergraph).
- **_edges: List[Edge]** — список рёбер (Edge из engine: source_node=graph_id, source_port, target_node=graph_id, target_port).
- **_in_edges_by_node: Dict[str, List[Edge]]** — для get_edges_in(graph_id).
- **_out_edges_by_node: Dict[str, List[Edge]]** — для get_edges_out(graph_id).
- **_exposed_inputs: List[Dict]** — записи {graph_id, port_name, name?}; name — опциональный ключ для inputs при run(workflow, inputs).
- **_exposed_outputs: List[Dict]** — записи {graph_id, port_name, name?}.
- **_execution_version: int** — инкремент при add_node, remove_node, add_edge, remove_edge, изменении exposed; для инвалидации кэша плана.
- **_workflow_id: str** — идентификатор воркфлоу (по умолчанию "workflow").
- **_workflow_kind: Optional[str]** — опциональный тег ("chain", "dag", "cyclic").
- **_metadata: Dict[str, Any]** — num_loop_steps, num_workflow_steps и прочие метаданные.
- **_node_trainable: Dict[str, bool]** — graph_id → trainable (по умолчанию True).

### 5.2 Методы добавления и удаления

**add_node(graph_id: str, hypergraph: Hypergraph) -> str**  
Добавить гиперграф как узел воркфлоу. graph_id должен быть уникален. Если graph_id уже есть — заменить или выбросить ValueError (по контракту: замена или запрет дубликата). Инкрементировать _execution_version. Вернуть graph_id.

**add_node(graph_id: str, config: Dict[str, Any], *, registry: Optional[BlockRegistry] = None, ref: Optional[str] = None) -> str**  
Построить гиперграф из конфига: если передан ref (путь к файлу), загрузить конфиг из ref; иначе использовать config. Вызвать Hypergraph.from_config(loaded_config, registry=registry). Добавить полученный гиперграф как узел с идентификатором graph_id (из config["graph_id"] или переданный graph_id). Инкрементировать _execution_version. Вернуть graph_id.

**remove_node(graph_id: str) -> None**  
Удалить узел и все рёбра, в которых он участвует; обновить _in_edges_by_node, _out_edges_by_node, удалить из _exposed_inputs/_exposed_outputs записи с этим graph_id. _execution_version += 1.

**add_edge(source_graph_id: str, source_port: str, target_graph_id: str, target_port: str) -> None**  
Создать Edge(source_node=source_graph_id, source_port=source_port, target_node=target_graph_id, target_port=target_port). Проверить: оба graph_id в _nodes; source_port входит в get_output_spec() гиперграфа source_graph_id; target_port входит в get_input_spec() гиперграфа target_graph_id. При несовпадении — ValueError. Добавить ребро в _edges и в _in_edges_by_node/_out_edges_by_node; _execution_version += 1. Дубликат ребра: идемпотентно игнорировать или ошибка — по контракту.

**remove_edge(source_graph_id: str, source_port: str, target_graph_id: str, target_port: str) -> None**  
Удалить соответствующее ребро; _execution_version += 1.

**expose_input(graph_id: str, port_name: str, name: Optional[str] = None) -> None**  
Добавить запись в _exposed_inputs: {graph_id, port_name, name?}. port_name должен быть во внешних входах гиперграфа (get_input_spec()). Не дублировать запись с тем же (graph_id, port_name).

**expose_output(graph_id: str, port_name: str, name: Optional[str] = None) -> None**  
Аналогично для _exposed_outputs; port_name — внешний выход гиперграфа (get_output_spec()).

### 5.3 Методы доступа (контракт для движка)

- **node_ids -> Set[str]** — return set(_nodes.keys()). Для движка «узлы» воркфлоу — это graph_id.
- **get_node(graph_id: str) -> Optional[Hypergraph]** — return _nodes.get(graph_id).
- **get_edges() -> List[Edge]** — return list(_edges).
- **get_edges_in(graph_id: str) -> List[Edge]** — return _in_edges_by_node.get(graph_id, []).
- **get_edges_out(graph_id: str) -> List[Edge]** — return _out_edges_by_node.get(graph_id, []).
- **get_input_spec(include_dtype: bool = False) -> List[Dict]** — обход _exposed_inputs; для каждой записи (graph_id, port_name, name) вернуть dict с **node_id** = graph_id (для совместимости с движком, который ожидает node_id в spec), port_name, name (и при include_dtype — dtype из гиперграфа.get_input_spec(include_dtype=True) по соответствующему порту). Движок формирует ключ буфера как (node_id, port_name) и внешний ключ inputs/outputs как name или "node_id:port_name"; под node_id на уровне воркфлоу подразумевается graph_id.
- **get_output_spec(include_dtype: bool = False) -> List[Dict]** — аналогично по _exposed_outputs (node_id = graph_id в каждой записи).
- **metadata -> Dict[str, Any]** — return dict(_metadata); движок читает num_loop_steps, num_workflow_steps оттуда.
- **execution_version** — для планировщика (инвалидация кэша).

Сигнатуры (псевдокод):

```python
class Workflow:
    def __init__(self, workflow_id: Optional[str] = None) -> None: ...

    @property
    def node_ids(self) -> Set[str]: ...
    def get_node(self, graph_id: str) -> Optional[Hypergraph]: ...
    def get_edges(self) -> List[Edge]: ...
    def get_edges_in(self, graph_id: str) -> List[Edge]: ...
    def get_edges_out(self, graph_id: str) -> List[Edge]: ...
    def get_input_spec(self, include_dtype: bool = False) -> List[Dict[str, Any]]: ...
    def get_output_spec(self, include_dtype: bool = False) -> List[Dict[str, Any]]: ...
    @property
    def metadata(self) -> Dict[str, Any]: ...
    @property
    def execution_version(self) -> int: ...

    def add_node(self, graph_id: str, hypergraph: Hypergraph) -> str: ...
    def add_node(self, graph_id: str, config: Dict[str, Any], *, registry: Optional[BlockRegistry] = None, ref: Optional[str] = None) -> str: ...
    def remove_node(self, graph_id: str) -> None: ...
    def add_edge(self, source_graph_id: str, source_port: str, target_graph_id: str, target_port: str) -> None: ...
    def remove_edge(self, source_graph_id: str, source_port: str, target_graph_id: str, target_port: str) -> None: ...
    def expose_input(self, graph_id: str, port_name: str, name: Optional[str] = None) -> None: ...
    def expose_output(self, graph_id: str, port_name: str, name: Optional[str] = None) -> None: ...
```

---

## 6. Разрешение ключей inputs/outputs и буферы

При run(workflow, inputs) ключи в словаре inputs — те же, что и для гиперграфа: **name** (если задан в expose_input) или канонический идентификатор **(graph_id, port_name)** или строка `"graph_id:port_name"`. Реализация движка (executor) при инициализации буфера по get_input_spec() получает для воркфлоу записи с полями graph_id, port_name, name; ключ в буфере — (graph_id, port_name); ключ во внешнем inputs — name или "graph_id:port_name". Аналогично для outputs по get_output_spec().

Буфер на уровне воркфлоу: ключ **(graph_id, port_name)** — то же, что (node_id, port_name) на уровне гиперграфа. Исполнитель не меняется: он обходит node_ids (для воркфлоу это graph_ids), для каждого id вызывает get_node(id).run(inputs_for_node), записывает выходы в буфер по (id, port_name). Единственное условие — get_node(id) возвращает объект с методом run(inputs) -> outputs. Hypergraph имеет run(inputs); значит, исполнение воркфлоу работает без изменений в executor.

---

## 7. Валидация воркфлоу

Валидатор должен проверять:

1. Все graph_id из рёбер и из _exposed_inputs/_exposed_outputs присутствуют в _nodes.
2. Для каждого ребра (source_graph_id, source_port, target_graph_id, target_port): source_port входит в **внешние выходы** гиперграфа source_graph_id (get_output_spec()); target_port входит в **внешние входы** гиперграфа target_graph_id (get_input_spec()).
3. Совместимость типов: тип выходного порта источника совместим с типом входного порта приёмника (если в get_output_spec/get_input_spec есть dtype — проверять по правилам совместимости).
4. Обязательные внешние входы гиперграфов-приёмников имеют входящее ребро воркфлоу или запись во внешних входах воркфлоу.
5. При наличии циклов между гиперграфами — предупреждение или проверка, что задано num_loop_steps / num_workflow_steps (в metadata или в опциях run).

Реализация: либо общий Validator принимает структуру и для проверки портов вызывает get_node(id).get_input_spec() и get_node(id).get_output_spec() (тогда для Workflow получаем внешние порты гиперграфа); либо отдельный WorkflowValidator обходит рёбра и exposed_* и проверяет по гиперграфам. Рекомендуется **единый валидатор** с абстракцией «структура с get_node(id) → узел с get_input_spec/get_output_spec».

---

## 8. Планировщик и циклы между гиперграфами

Планировщик (ExecutionPlanner) строит план по графу зависимостей: вершины = node_ids (для воркфлоу = graph_ids), дуги = направление рёбер (source → target). Для **DAG** — топологическая сортировка. Для **графа с циклом** — разбиение на начальную фазу, циклическую фазу (SCC с циклом) и конечную фазу; циклическая фаза выполняется K раз (num_loop_steps). Логика та же, что и для гиперграфа задачи (Phase 2); планировщик не зависит от типа узла, только от списка id и рёбер. Поэтому **тот же планировщик** применим к воркфлоу: build_plan(workflow) возвращает план (список шагов "node" или "cycle" с порядком graph_id). Кэш плана привязан к execution_version структуры; при изменении воркфлоу кэш инвалидируется.

---

## 9. Контракт run и интеграция с движком

**run(workflow, inputs, **options) -> outputs**

Реализация: метод Workflow.run(inputs, **options) делегирует в engine.run(self, inputs, **options). То есть вызывается та же функция run(structure, inputs, ...), что и для гиперграфа задачи; structure = workflow. Исполнитель обходит план, для каждого graph_id вызывает node = workflow.get_node(graph_id), node.run(inputs_for_graph), записывает выходы в буфер. node — это Hypergraph, его run(inputs) выполнит весь внутренний граф (включая циклы и agent_loop внутри гиперграфа). Опции (training, device, num_loop_steps, callbacks, dry_run, validate_before) обрабатываются движком; num_loop_steps для цикла **между** гиперграфами берётся из workflow.metadata или из options.

**Проброс опций в гиперграфы:** при вызове node.run(inputs_for_graph) движок может передавать опции в run гиперграфа (training, device уже в фазе 2/3 пробрасываются). Для воркфлоу опции run(workflow, inputs, device=..., training=...) применяются к каждому гиперграфу-узлу (to(device), training в run(hypergraph, inputs, training=..., ...)).

Сигнатура метода Workflow:

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
    from yggdrasill.engine.executor import run as _run
    return _run(self, inputs, training=training, num_loop_steps=num_loop_steps,
                device=device, callbacks=callbacks, dry_run=dry_run,
                validate_before=validate_before, **kwargs)
```

---

## 10. from_config и to_config

### 10.1 Структура конфига воркфлоу

- **schema_version** — строка, например "1.0".
- **workflow_id** — идентификатор воркфлоу.
- **graphs** — список узлов. Каждый элемент: **graph_id** (обязательно), **config** (полный конфиг гиперграфа по 03 §9.1) или **ref** (путь к файлу конфига гиперграфа). Опционально: trainable.
- **edges** — список рёбер: source_graph, source_port, target_graph, target_port.
- **exposed_inputs** — список {graph_id, port_name, name?}.
- **exposed_outputs** — список {graph_id, port_name, name?}.
- **workflow_kind** — опционально.
- **metadata** — опционально (num_loop_steps, num_workflow_steps и т.д.).

### 10.2 from_config(config, registry?, validate?)

- Разобрать config["graphs"]: для каждого элемента с graph_id и config или ref построить Hypergraph (если ref — загрузить конфиг из файла, затем Hypergraph.from_config(loaded, registry)); добавить в воркфлоу add_node(graph_id, hypergraph) или add_node(graph_id, config, registry).
- Разобрать config["edges"]: для каждого add_edge(source_graph, source_port, target_graph, target_port).
- Восстановить exposed_inputs, exposed_outputs из config["exposed_inputs"], config["exposed_outputs"].
- Установить workflow_id, workflow_kind, metadata из config.
- Если validate=True — вызвать валидатор воркфлоу; при ошибках — исключение или возврат с флагом (по контракту).
- Вернуть экземпляр Workflow.

### 10.3 to_config()

- Вернуть словарь: schema_version, workflow_id, graphs (для каждого graph_id — {graph_id, config: hypergraph.to_config()} или ref, если гиперграф был загружен по ref), edges, exposed_inputs, exposed_outputs, workflow_kind, metadata. При сохранении по ref в конфиге может храниться только ref, а не вложенный config; to_config тогда возвращает ref для такого узла (если реализация хранит источник конфига).

---

## 11. state_dict, load_state_dict, to(device), trainable

- **state_dict()** — агрегат по graph_id: {graph_id: hypergraph.state_dict() for graph_id, hypergraph in _nodes.items()}. При общем ref (один гиперграф в нескольких «логических» узлах) в фазе 6 обычно один graph_id на один экземпляр; если бы один и тот же экземпляр был представлен двумя graph_id, хранить один раз по ключу (канон SERIALIZATION §4.4 для воркфлоу).
- **load_state_dict(state: Dict, strict: bool = True)** — для каждого graph_id в _nodes вызвать hypergraph.load_state_dict(state.get(graph_id, {}), strict=...). При strict проверить, что ключи state совпадают с graph_ids воркфлоу.
- **to(device)** — для каждого гиперграфа в _nodes вызвать hypergraph.to(device); return self.
- **trainable_parameters()** — итератор по параметрам только тех гиперграфов, для которых _node_trainable.get(graph_id, True) == True; для каждого гиперграфа hypergraph.trainable_parameters() (если есть).
- **set_trainable(graph_id: str, trainable: bool)** — установить _node_trainable[graph_id] = trainable.

---

## 12. Сериализация воркфлоу (save/load)

По канону (04 §8, SERIALIZATION §2): конфиг воркфлоу + чекпоинт как совокупность чекпоинтов гиперграфов.

- **save(path)** — записать в path (директория): config.json (или config_filename) с to_config(); чекпоинт — либо один файл checkpoint.pt с state_dict() воркфлоу (словарь graph_id → state_dict гиперграфа), либо поддиректории по graph_id, в каждой — конфиг и чекпоинт гиперграфа (по правилам фазы 5). Рекомендуется единый checkpoint.pt со словарём graph_id → state_dict для простоты.
- **save_config(path, *, filename="config.json")** — записать только to_config().
- **save_checkpoint(path, *, filename="checkpoint.pt")** — записать только state_dict() воркфлоу.
- **load(path, registry?, validate?, load_checkpoint=True, ...)** — класс-метод: прочитать конфиг из path, Workflow.from_config(config, registry, validate); если load_checkpoint — прочитать чекпоинт и вызвать workflow.load_state_dict(loaded); вернуть Workflow.
- **load_config(path, ...)** — только from_config из файла конфига в path.
- **load_from_checkpoint(path, ...)** — метод экземпляра: прочитать чекпоинт из path и self.load_state_dict(loaded).

При **общем ref** (несколько записей в graphs с одним ref на один и тот же гиперграф): при from_config можно строить один экземпляр Hypergraph и подставлять его под разными graph_id (тогда в _nodes два ключа указывают на один объект) или хранить один раз и при run использовать один и тот же экземпляр. Канон 04 §8.2: один экземпляр при общем ref; чекпоинт воркфлоу тогда содержит один чекпоинт для этого ref. Реализация: при from_config кэшировать построенные гиперграфы по ref; при совпадении ref подставлять тот же экземпляр для нескольких graph_id (при необходимости graph_id различаются для рёбер и exposed, но объект один).

---

## 13. infer_exposed_ports и auto_connect (опционально)

### 13.1 infer_exposed_ports()

**infer_exposed_ports()** — установить _exposed_inputs и _exposed_outputs по правилу: внешний вход воркфлоу = порт гиперграфа (graph_id, port_name), на который **нет входящего ребра воркфлоу** от другого гиперграфа; внешний выход воркфлоу = порт гиперграфа, от которого **нет исходящего ребра воркфлоу** к другому гиперграфу. Обход всех гиперграфов и их get_input_spec()/get_output_spec(); для каждого порта проверка наличия входящего/исходящего ребра воркфлоу. Вызов перезаписывает текущие списки exposed.

### 13.2 auto_connect между гиперграфами

По аналогии с уровнем абстрактных узлов-задач (автосвязывание ролей) воркфлоу может поддерживать **автоматическое предложение или создание связей между гиперграфами** на основе их внешних портов. Это не обязательная часть минимальной реализации, но уровень спроектирован так, чтобы auto_connect было легко добавить.

**Идея:**

- Каждый гиперграф-узел предоставляет `get_input_spec(include_dtype=True)` и `get_output_spec(include_dtype=True)` — имена и типы внешних портов.
- На уровне воркфлоу можно реализовать утилиту вида:

  ```python
  def suggest_auto_edges(workflow: Workflow) -> List[Tuple[str, str, str, str]]:
      ...

  def apply_auto_connect(workflow: Workflow) -> None:
      for (src_g, src_p, dst_g, dst_p) in suggest_auto_edges(workflow):
          workflow.add_edge(src_g, src_p, dst_g, dst_p)
  ```

- Внутри `suggest_auto_edges` перебираются пары (выходной порт одного гиперграфа, входной порт другого) и по **правилам совместимости** строится список кандидатов:
  - по совпадению имени порта (`"image" → "image"`, `"latent" → "latent"` и т.п.);
  - по типу (`dtype`: `IMAGE`, `TEXT`, `LATENT` …);
  - при наличии семантических метаданных графа (например, `workflow_kind` или role-like теги) — по шаблонам («выход text2img.image идёт в upscale.image»).

**Варианты поведения:**

- **Режим предложения** — `suggest_auto_edges` возвращает список кандидатов; пользователь/верхний уровень решает, какие из них принять (например, в UI).
- **Режим автосвязывания** — `apply_auto_connect(workflow, strategy=...)` сразу вызывает `add_edge` для каждого кандидата согласно выбранной стратегии (например, «один источник → один лучший приёмник по совпадению имени и dtype»).

Ключевой момент: auto_connect **не меняет базовый контракт** движка и структур — он реализуется на уровне `Workflow` как дополнительный слой, опирающийся на уже существующий контракт внешних портов гиперграфов (get_input_spec/get_output_spec). Ядро (Validator, Planner, Executor) остаётся неизменным.

---

## 14. Граничные случаи и тонкости

- **Один узел (вырожденный воркфлоу):** воркфлоу из одного гиперграфа допустим. План = один шаг; run(workflow, inputs) по сути делегирует run(hypergraph, inputs). Внешние входы/выходы воркфлоу = внешние входы/выходы этого гиперграфа.
- **Пустые exposed_inputs/exposed_outputs:** если не объявлены, run не сможет сопоставить inputs с портами; при infer_exposed_ports() они заполнятся. Для пустого воркфлоу (ноль узлов) to_config и state_dict возвращают пустые структуры; run с пустым воркфлоу не имеет смысла (ошибка или пустой outputs).
- **Цикл между двумя гиперграфами:** A → B → A. Планировщик строит фазу цикла [A, B]; выполняется K раз; num_loop_steps из metadata или options. Буферы обновляются после каждого вызова A и B; на следующей итерации B читает новые данные от A.
- **Опциональные входы гиперграфа:** гиперграф может иметь опциональный внешний вход; на него не обязано вести ребро воркфлоу и не обязано подаваться значение в inputs; при сборе входов для run(hypergraph, inputs) передаём только те ключи, для которых есть данные в буфере или в inputs воркфлоу.
- **Ключи inputs/outputs:** единообразие с гиперграфом задачи: name или "graph_id:port_name". Движок при get_input_spec() получает записи с graph_id, port_name, name; формирует ключ для пользователя так же, как для node_id на уровне задачи.
- **Ошибки выполнения:** при сбое одного гиперграфа (исключение, таймаут) политика не задаётся каноном; реализация может прерывать run или обрабатывать по-другому (04 §11.6).

---

## 15. Тесты

**Расположение:** tests/workflow/ (или tests/engine/test_workflow.py).

**Сценарии:**

1. **Добавление узлов и рёбер:** add_node(graph_id, hypergraph), add_edge(g1, p1, g2, p2); get_node, get_edges_in/out, get_input_spec, get_output_spec возвращают ожидаемое; execution_version растёт при изменениях.
2. **from_config / to_config roundtrip:** workflow.from_config(config, registry); workflow.to_config() эквивалентен исходному конфигу (сравнение ключей и значений для graphs, edges, exposed_*).
3. **run цепочки из двух гиперграфов:** два гиперграфа (заглушки) A и B: A(out) → B(in). expose_input(A, in_a), expose_output(B, out_b). run(workflow, {in_a: x}) → проверка outputs[out_b] совпадает с результатом последовательного run(A, {in_a: x}) и run(B, {in_b: A_output}).
4. **run цепочки из трёх гиперграфов:** A → B → C; один внешний вход, один внешний выход; проверка результата run(workflow, inputs).
5. **Цикл между гиперграфами:** A → B → A; num_loop_steps=2; проверка, что A и B вызваны по два раза и выход воркфлоу согласован с ожиданием (для детерминированных заглушек).
6. **Вырожденный воркфлоу (один гиперграф):** add_node("g", hypergraph); expose_input("g", pin), expose_output("g", pout); run(workflow, inputs) == hypergraph.run(inputs).
7. **state_dict / load_state_dict:** после run сохранить state_dict воркфлоу; изменить состояние одного гиперграфа; load_state_dict сохранённого; снова run — результат как до изменения (воспроизводимость).
8. **save / load воркфлоу:** workflow.save(dir); w2 = Workflow.load(dir, registry=...); w2.to_config() эквивалентен workflow.to_config(); run(w2, inputs) даёт тот же результат, что run(workflow, inputs) (при тех же чекпоинтах).
9. **Валидация:** невалидный воркфлоу (ребро в несуществующий graph_id, порт не из get_output_spec/get_input_spec) — validate возвращает ошибки; run(..., validate_before=True) выбрасывает исключение.
10. **to(device), set_trainable:** вызов to(device) не падает; set_trainable(graph_id, False); trainable_parameters() не включает параметры этого графа (если реализация различает).

---

## 16. Чек-лист реализации

- [ ] Пакет yggdrasill/workflow: __init__.py, structure.py (Workflow), при необходимости io.py.
- [ ] Workflow: _nodes, _edges, _in_edges_by_node, _out_edges_by_node, _exposed_inputs, _exposed_outputs, _execution_version, _workflow_id, _workflow_kind, _metadata, _node_trainable.
- [ ] add_node(graph_id, hypergraph), add_node(graph_id, config, registry?, ref?); remove_node(graph_id).
- [ ] add_edge(source_graph_id, source_port, target_graph_id, target_port); remove_edge(...).
- [ ] expose_input(graph_id, port_name, name?), expose_output(graph_id, port_name, name?).
- [ ] node_ids, get_node(graph_id), get_edges(), get_edges_in(graph_id), get_edges_out(graph_id), get_input_spec(include_dtype?), get_output_spec(include_dtype?), metadata, execution_version.
- [ ] Валидация воркфлоу: порты по get_node(id).get_input_spec()/get_output_spec(); общий валидатор или WorkflowValidator.
- [ ] run(workflow, inputs, **options) — делегирование в engine.run(workflow, inputs, ...); тот же executor.
- [ ] Планировщик и циклы между гиперграфами: num_loop_steps из metadata/options.
- [ ] from_config(config, registry?, validate?), to_config().
- [ ] state_dict(), load_state_dict(state); to(device), trainable_parameters(), set_trainable(graph_id, bool).
- [ ] save(path), save_config(path), save_checkpoint(path); load(path, ...), load_config(path, ...), load_from_checkpoint(path).
- [ ] infer_exposed_ports() (опционально).
- [ ] (Опционально) auto_connect на уровне воркфлоу: suggest_auto_edges(workflow) по get_input_spec/get_output_spec гиперграфов и apply_auto_connect(workflow) поверх add_edge.
- [ ] Тесты: add_node/add_edge, from_config/to_config roundtrip, run цепочки из 2–3 гиперграфов, цикл между гиперграфами, вырожденный воркфлоу, state_dict/load_state_dict, save/load, валидация.
- [ ] Обновить IMPLEMENTATION_PLAN и README ссылкой на PHASE_6_WORKFLOW.md.

---

## 17. Связь с каноном и другими фазами

| Документ / фаза | Связь |
|-----------------|--------|
| **04_WORKFLOW.md** | Воркфлоу = гиперграф гиперграфов; узлы = гиперграфы задач; рёбра = внешние порты; тот же движок; циклы между гиперграфами; контракт run; сериализация. |
| **03_TASK_HYPERGRAPH.md** | Гиперграф задачи имеет get_input_spec, get_output_spec, run — контракт «узла» для воркфлоу. |
| **HYPERGRAPH_ENGINE.md** | Движок универсален: узлы с run(inputs)→outputs; применяется к воркфлоу без смены ядра. |
| **SERIALIZATION.md** | Конфиг воркфлоу + чекпоинт = агрегат чекпоинтов гиперграфов; загрузка только чекпоинтов в уже собранный воркфлоу. |
| **Фаза 2** | Edge, Validator, Planner, Executor — те же для воркфлоу; структура Workflow реализует контракт (node_ids, get_node, get_edges, get_input_spec, get_output_spec). |
| **Фаза 3** | Hypergraph.run, get_input_spec, get_output_spec — вызываются движком для каждого «узла» воркфлоу. |
| **Фаза 5** | Формат save/load воркфлоу по аналогии с гиперграфом; чекпоинт воркфлоу — словарь graph_id → state_dict гиперграфа. |
| **Фаза 7** | Стадия опирается на воркфлоу как на узел (state_input_map, state_output_map, run(stage, state) → state). |

---

**Итог.** Документ задаёт полную спецификацию **фазы 6**: воркфлоу как гиперграф гиперграфов (узлы = гиперграфы задач, рёбра = связи между внешними портами), **без смены ядра движка** — тот же Validator, Planner, Executor; контракт структуры (node_ids, get_node, get_edges, get_input_spec, get_output_spec) реализуется классом Workflow; поддержка циклов между гиперграфами (итеративная схема, num_loop_steps); from_config, to_config, state_dict, load_state_dict, save/load; тесты и граничные случаи. Реализация и уровень стадии опираются на эту спецификацию.
