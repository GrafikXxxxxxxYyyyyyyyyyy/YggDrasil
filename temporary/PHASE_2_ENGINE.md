# Фаза 2. Гиперграфовый движок — полный технический план

Детальный технический план по **второй фазе**: гиперграфовый движок и минимальная исполняемая структура (рёбра, контейнер узлов и рёбер, валидатор, планировщик, буферы, исполнитель). Цель — чтобы после реализации **с первого раза** можно было собрать граф из узлов (AbstractGraphNode), соединить их рёбрами, объявить внешние входы/выходы и выполнить **run(hypergraph, inputs) → outputs**. Документ опирается на канон (01, 03, HYPERGRAPH_ENGINE), план реализации и референс (outdated_1).

**Связь с планом:** [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — Фаза 2.

**Канон:** [documentation/docs/HYPERGRAPH_ENGINE.md](../documentation/docs/HYPERGRAPH_ENGINE.md), [documentation/docs/01_FOUNDATION.md](../documentation/docs/01_FOUNDATION.md) §5–6, [documentation/docs/03_TASK_HYPERGRAPH.md](../documentation/docs/03_TASK_HYPERGRAPH.md) §5–7.

**Референс:** reference/outdated_1/foundation/graph.py, reference/outdated_1/executor/run.py.

**Язык:** русский.

---

## 1. Цель фазы 2

Реализовать **гиперграфовый движок** и **минимальную исполняемую структуру** так, чтобы:

- Можно было **хранить** граф: узлы (node_id → узел-задача, объект Block+Node, наследующий AbstractGraphNode и AbstractBaseBlock), рёбра (источник_узел, источник_порт → приёмник_узел, приёмник_порт), внешние входы/выходы.
- **Валидатор** проверял структуру перед выполнением (узлы и порты в рёбрах существуют, типы совместимы, обязательные входы покрыты).
- **Планировщик** строил план выполнения: для DAG — топологический порядок; для графа с циклом — разбиение на фазы с итеративной частью (K шагов).
- **Исполнитель** по плану: инициализировал буферы из внешних входов, обходил узлы, для каждого узла собирал входы из буферов, вызывал **node.run(inputs)**, записывал выходы в буферы, по окончании собирал внешние выходы.
- Контракт: **run(hypergraph, inputs, **options) → outputs** с устойчивыми ключами (name или "node_id:port_name").

**Результат фазы 2:** фреймворк «оживает»: цепочка из 2–3 узлов (например IdentityBlock) с рёбрами и объявленными входами/выходами даёт воспроизводимый вызов run и корректные outputs. Поддерживаются DAG и один итеративный цикл (K шагов). Agent_loop и сериализация гиперграфа — фаза 3.

---

## 2. Зависимости

- **Фаза 0:** структура репозитория (yggdrasill/, tests/), pyproject.toml, pytest. См. [PHASE_0_STRUCTURE.md](PHASE_0_STRUCTURE.md).
- **Фаза 1:** Port, AbstractBaseBlock, AbstractGraphNode, BlockRegistry; узлы умеют declare_ports(), run(inputs)→outputs; порты имеют direction (IN/OUT), compatible_with(). См. [PHASE_1_FOUNDATION.md](PHASE_1_FOUNDATION.md).

Движок **не знает** о конкретных типах блоков (диффузия, LLM); он только обходит узлы и вызывает run.

---

## 3. Что входит в фазу 2 и что нет

| Входит в фазу 2 | Не входит (фаза 3 и далее) |
|-----------------|----------------------------|
| Edge (ребро арности 2: source_node, source_port, target_node, target_port). | Гиперрёбра арности > 2 (add_hyperedge(ends)); можно заменить несколькими Edge. |
| Контейнер структуры: узлы, рёбра, exposed_inputs, exposed_outputs, execution_version; add_node, add_edge, expose_input, expose_output; get_node, get_edges, get_edges_in/out, get_input_spec, get_output_spec. | from_config / to_config, загрузка из реестра по block_type (остаётся на фазу 3). |
| Валидатор: проверки по канону §5 (03), результат ValidationResult (errors, warnings). | Сериализация конфига и чекпоинта гиперграфа. |
| Планировщик: топологический порядок, SCC (Tarjan), план для DAG и для одного цикла (итеративная фаза K раз). | Несколько циклов в одном графе; agent_loop (можно заглушить или упростить). |
| Буферы: ключ (node_id, port_name), инициализация из inputs, запись после run узла, агрегация при нескольких входящих рёбрах на один порт (по Port.aggregation). | Автосвязывание (auto_connect), graph_kind, шаблоны. |
| Исполнитель: run(hypergraph, inputs, **options) → outputs; опции num_loop_steps, training, device, dry_run, callbacks. | |

---

## 4. Размещение кода

```
yggdrasill/
  foundation/          # уже есть (фаза 1)
  engine/              # новый пакет фазы 2
    __init__.py        # экспорт Edge, Hypergraph, Validator, Planner, Executor, run
    edge.py            # Edge (source_node, source_port, target_node, target_port)
    structure.py       # Hypergraph — узлы, рёбра, exposed, execution_version, add_*, expose_*, get_*
    validator.py       # Validator, ValidationResult
    planner.py         # ExecutionPlanner — build_plan(hypergraph), кэш по execution_version
    buffers.py         # EdgeBuffers — буфер (node_id, port) → value; init, write, read, aggregate
    executor.py        # run(hypergraph, inputs, **options) → outputs

tests/
  engine/
    __init__.py
    test_edge.py
    test_structure.py
    test_validator.py
    test_planner.py
    test_executor.py
    helpers.py         # минимальные графы для тестов (цепочка A→B→C, цикл из двух узлов)
```

Импорт после фазы 2:

```python
from yggdrasill.engine import Edge, Hypergraph, Validator, run
# или
from yggdrasill.engine.executor import run
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.engine.edge import Edge
```

---

## 5. Edge — представление ребра

Ребро соединяет один выходной порт узла с одним входным портом другого узла (арность 2). Гиперрёбра арности > 2 в фазе 2 не реализуем; несколько рёбер от одного порта к разным приёмникам допустимы.

**Файл:** `yggdrasill/engine/edge.py`.

**Атрибуты:**

- `source_node: str` — идентификатор узла-источника.
- `source_port: str` — имя выходного порта источника.
- `target_node: str` — идентификатор узла-приёмника.
- `target_port: str` — имя входного порта приёмника.

**Требования:**

- Иммутабельность: после создания поля не меняются (например, `@dataclass(frozen=True)` или `__slots__` без сеттеров).
- Нормализация: идентификаторы узлов и портов — строки, при создании можно вызывать `.strip()` (по аналогии с Phase 1).
- Равенство: два ребра равны, если совпадают все четыре поля (для дедупликации и проверки при add_edge).

**Сигнатура (пример):**

```python
@dataclass(frozen=True)
class Edge:
    source_node: str
    source_port: str
    target_node: str
    target_port: str

    def __post_init__(self) -> None:
        # опционально: проверка непустых строк
        ...
```

**Тесты (tests/engine/test_edge.py):**

- Создание Edge, равенство и неравенство, хэш (если frozen).
- Нельзя создать ребро с пустым source_node/target_node (если валидация в __post_init__).

---

## 6. Хранилище структуры — Hypergraph

Контейнер, который хранит узлы, рёбра и списки внешних входов/выходов. Движок читает из него данные для валидации, планирования и выполнения. В фазе 2 узел добавляется как готовый экземпляр **узла-задачи** (объект Block+Node); создание по block_type через реестр — фаза 3.

**Файл:** `yggdrasill/engine/structure.py`.

**Внутреннее состояние:**

- `_nodes: Dict[str, ...]` — node_id → узел-задача (объект, наследующий AbstractGraphNode и AbstractBaseBlock; тип для аннотаций — абстрактный базовый класс узла или протокол).
- `_edges: List[Edge]` — список рёбер (или set для быстрой проверки дубликатов).
- `_in_edges_by_node: Dict[str, List[Edge]]` — для быстрого get_edges_in(node_id).
- `_out_edges_by_node: Dict[str, List[Edge]]` — для быстрого get_edges_out(node_id).
- `_exposed_inputs: List[Dict]` — список записей `{node_id, port_name, name?}` (name — опциональный ключ для inputs).
- `_exposed_outputs: List[Dict]` — список записей `{node_id, port_name, name?}`.
- `_execution_version: int` — увеличивается при add_node, remove_node, add_edge, remove_edge, изменении exposed; используется для инвалидации кэша плана.
- `metadata: Dict[str, Any]` — опционально (например `num_loop_steps` по умолчанию).

**Методы добавления/удаления:**

- `add_node(node_id: str, node: AbstractGraphNode) -> None`  
  Добавить узел. Если node_id уже есть — заменить или выбросить (по контракту: в референсе — замена). Инкрементировать _execution_version. Проверять, что node_id и имена портов узла не пустые.
- `remove_node(node_id: str) -> None`  
  Удалить узел и все рёбра, где он источник или приёмник; удалить записи в _exposed_inputs/_exposed_outputs для этого node_id; инкрементировать _execution_version.
- `add_edge(edge: Edge) -> None`  
  Проверить: source_node и target_node в _nodes; source_port — выходной порт узла-источника (порт с direction OUT), target_port — входной порт узла-приёмника (direction IN); типы портов совместимы (source_port.compatible_with(target_port)). Если ребро уже есть — не дублировать (идемпотентно). Добавить в _edges и в _in_edges_by_node/_out_edges_by_node; _execution_version += 1. Получение порта по имени: через get_input_ports()/get_output_ports() блока найти порт с port.name == port_name; при отсутствии в Phase 1 метода get_port(name) реализовать хелпер или добавить get_port в блок (см. PHASE_1_FOUNDATION.md).
- `remove_edge(edge: Edge) -> None`  
  Удалить ребро из списков; _execution_version += 1.
- `expose_input(node_id: str, port_name: str, name: Optional[str] = None) -> None`  
  Узел должен существовать; port_name — входной порт. Добавить запись в _exposed_inputs: `{node_id, port_name, name}` (name может быть None).
- `expose_output(node_id: str, port_name: str, name: Optional[str] = None) -> None`  
  Аналогично для выходного порта.

**Методы чтения (для движка):**

- `node_ids -> Set[str]` или `list(_nodes.keys())`.
- `get_node(node_id: str) -> Optional[AbstractGraphNode]`.
- `get_edges() -> List[Edge]`.
- `get_edges_in(node_id: str) -> List[Edge]`.
- `get_edges_out(node_id: str) -> List[Edge]`.
- `get_input_spec() -> List[Dict]` — копия _exposed_inputs (для ключей run: name или "node_id:port_name").
- `get_output_spec() -> List[Dict]` — копия _exposed_outputs.
- `execution_version -> int` (property).

**Ключи для inputs/outputs (соглашение):**

- Если в записи exposed задан `name` — ключ в словаре inputs/outputs = `name`.
- Иначе — ключ = `"node_id:port_name"` (устойчивый канонический вид). Реализация: вспомогательная функция `_input_key(entry)` / `_output_key(entry)` как в reference/outdated_1/executor/run.py.

**Тесты (test_structure.py):**

- add_node, get_node, remove_node; после remove_node рёбра с этим node_id удалены.
- add_edge: успех при корректных портах; исключение при несуществующем узле или несовместимых портах; идемпотентность при повторном add_edge того же ребра.
- expose_input/expose_output: записи в get_input_spec/get_output_spec; ключ по name или "node_id:port_name".
- execution_version увеличивается после add_node, add_edge, remove_node, remove_edge.

---

## 7. Валидатор

Проверяет, что **любая исполняемая структура** (гиперграф задачи, воркфлоу, стадия и т.д.) **реализуема** перед выполнением: все ссылки на узлы и порты корректны, типы на концах рёбер совместимы, обязательные входы покрыты. Валидатор обобщён и работает с любой структурой, реализующей единый интерфейс.

**Файл:** `yggdrasill/engine/validator.py`.

**Вход:** объект структуры (далее `structure`), реализующий общий контракт:

- `structure.node_ids` — множество идентификаторов узлов на данном уровне (для гиперграфа задачи — node_id узлов-задач; для воркфлоу — graph_id гиперграфов и т.д.).
- `structure.get_node(id)` — вернуть объект узла по идентификатору (узел-задача, гиперграф, воркфлоу и т.п.).
- `structure.get_edges()` — список рёбер с полями `source_node`, `source_port`, `target_node`, `target_port` (Edge из engine).
- `structure.get_input_spec(include_dtype: bool = False)` — список внешних входов структуры: записи с полями как минимум `node_id`, `port_name`, опционально `name` и `dtype`.
- `structure.get_output_spec(include_dtype: bool = False)` — аналогично для внешних выходов.

Узлы, которые возвращает `get_node(id)`, сами предоставляют **списки своих портов** и/или внешних портов (на уровне задачи — порты узлов-задач через declare_ports; на уровне воркфлоу — внешние порты гиперграфов через get_input_spec/get_output_spec самого гиперграфа). Валидатор при необходимости может вызывать `node.get_input_spec()` / `node.get_output_spec()` для уточнения списка портов узла.

**Выход:** объект `ValidationResult` с полями:

- `errors: List[str]` — критические ошибки (выполнение не должно стартовать).
- `warnings: List[str]` — предупреждения (необязательные порты без входящего ребра, циклы без num_loop_steps в metadata и т.п.).

**Проверки (обязательные):**

1. Все идентификаторы из рёбер (source_node, target_node) присутствуют в `structure.node_ids`.
2. Все port_name в рёбрах соответствуют объявленным портам на концах рёбер: source_port — выход источника, target_port — вход приёмника (для уровня гиперграфа задачи — порты узлов-задач; для уровня воркфлоу — внешние порты гиперграфов). Источник списка портов задаётся либо портами узлов (declare_ports), либо внешними портами узлов (get_input_spec/get_output_spec) в зависимости от уровня.
3. Типы портов: для каждого ребра тип выходного порта источника совместим с типом входного порта приёмника (по правилам Port / dtype; например, `PortType.ANY` совместим со всем, остальные — по совпадению или подтипизации).
4. Записи в exposed_inputs и exposed_outputs структуры: node_id существует, port_name — объявленный внешний вход/выход соответствующего узла на данном уровне.
5. Обязательные входные порты: у каждого узла каждый обязательный (non-optional) входной порт имеет хотя бы одно входящее ребро или запись во внешних входах структуры (порт будет заполнен из внешнего входа).

**Опционально:**

- Связность: от внешних входов достижимы узлы внешних выходов (можно предупреждение, а не ошибка).
- Циклы: если структура содержит цикл (SCC размера > 1 или self-loop), в metadata может быть num_loop_steps; иначе — предупреждение.

**Сигнатура:**

```python
def validate(structure: Any) -> ValidationResult: ...
```

Где `structure` — Hypergraph, Workflow или другая структура, реализующая описанный интерфейс. Для ясности можно определить протокол `ExecutableStructure` с этими методами и использовать его в аннотациях типов.

**Тесты (test_validator.py):**

- Гиперграф задачи без ошибок — errors пустой.
- Воркфлоу без ошибок — errors пустой (использует тот же валидатор).
- Несуществующий node_id в ребре — ошибка.
- Несуществующий port_name — ошибка.
- Несовместимые типы портов — ошибка.
- Обязательный вход без входящего ребра и без внешнего входа — ошибка.
- Опциональный вход без входящего ребра — допустимо (или предупреждение).

---

## 8. Планировщик (Execution Planner)

Строит **план выполнения**: последовательность шагов. Каждый шаг — либо один узел (DAG), либо цикл (множество узлов, выполняемых K раз подряд). План кэшируется по паре (id(hypergraph), execution_version); при изменении версии кэш инвалидируется.

**Файл:** `yggdrasill/engine/planner.py`.

**Алгоритм (по канону и reference run.py):**

1. Построить ориентированный граф по рёбрам: вершины = node_ids, дуга (source_node → target_node) для каждого Edge.
2. **Tarjan** — разбить на сильно связные компоненты (SCC). Порядок SCC — обратный топологический (стоки первыми).
3. Построить DAG по SCC: каждая SCC — вершина; ребро между SCC A → B, если есть ребро из любого узла A в любой узел B (A ≠ B).
4. Топологическая сортировка этого DAG (Kahn или DFS) — порядок SCC: «источники» (без входящих из других SCC) первыми.
5. Для каждой SCC:
   - Если SCC из одного узла и у него нет self-loop (ребро node → node) — шаг плана: `("node", node_id)`.
   - Если SCC из нескольких узлов или один узел с self-loop — шаг плана: `("cycle", (representative_id, set_of_node_ids))`. Внутри цикла узлы выполняются в детерминированном порядке (например, sorted(node_ids)).

**Формат плана:** `List[Tuple[str, Any]]`:

- `("node", node_id: str)` — выполнить один раз узел node_id.
- `("cycle", (representative: str, comp: Set[str]))` — выполнить K раз (K из options) все узлы из comp в порядке sorted(comp).

**Кэш:** глобальный словарь или атрибут планировщика: ключ — (id(hypergraph), hypergraph.execution_version), значение — план. При вызове build_plan(hypergraph) если версия совпадает с закэшированной — вернуть кэш; иначе пересчитать и сохранить.

**Сигнатура:**

```python
def build_plan(hypergraph: Hypergraph) -> List[Tuple[str, Any]]: ...
```

**Тесты (test_planner.py):**

- Цепочка A → B → C: план из трёх шагов ("node", A), ("node", B), ("node", C) в этом порядке.
- Два узла A ↔ B (цикл): один шаг ("cycle", (rep, {A, B})); порядок узлов внутри цикла — sorted.
- Один узел без self-loop: один шаг ("node", node_id).
- Один узел с self-loop: один шаг ("cycle", (node_id, {node_id})).
- Кэш: два вызова build_plan для одного и того же графа без изменений — один и тот же объект плана; после add_edge — новый план (другая версия).

---

## 9. Буферы (Edge Buffers)

В рамках одного run исполнитель хранит **буфер** — отображение (node_id, port_name) → value. Это не отдельный публичный класс в минимальной реализации: буфер может быть обычным словарём внутри executor. Спецификация поведения:

- **Инициализация:** для каждой записи в get_input_spec() положить в буфер значение из словаря inputs по ключу (name или "node_id:port_name"). Ключ в буфере — (node_id, port_name).
- **После run(node_id):** для каждого выходного порта узла и каждого исходящего ребра записать outputs[port_name] в буфер по ключу (target_node, target_port). Если один выход раздаётся по нескольким рёбрам — несколько записей в буфер.
- **Несколько входящих рёбер на один порт:** собрать все значения по входящим рёбрам и применить **агрегацию** (по Port.aggregation: SINGLE, CONCAT, SUM, FIRST, DICT). Если в фазе 2 упростить — допустить только один входящий ребро на порт или агрегацию FIRST/CONCAT по умолчанию.
- **Сбор выходов:** по get_output_spec() для каждой записи взять из буфера значение по (node_id, port_name) и записать в результат по ключу (name или "node_id:port_name").

Отдельный модуль `buffers.py` может содержать класс EdgeBuffers с методами init_from_inputs(spec, inputs), write(node_id, port_name, value), read(node_id, port_name), aggregate(node_id, port_name, values) — или вся логика в executor. Для «запуска с первого раза» достаточно словаря внутри executor и явного описания в документе.

---

## 10. Исполнитель (Executor) и run

Единая точка входа выполнения: **run(hypergraph, inputs, **options) → outputs**.

**Файл:** `yggdrasill/engine/executor.py`.

**Сигнатура:**

```python
def run(
    hypergraph: Hypergraph,
    inputs: Dict[str, Any],
    *,
    training: bool = False,
    num_loop_steps: Optional[int] = None,
    device: Optional[Any] = None,
    callbacks: Optional[List[Callable[..., None]]] = None,
    dry_run: bool = False,
    validate_before: bool = True,
) -> Dict[str, Any]: ...
```

**Поведение:**

1. **Валидация (опционально):** если validate_before — вызвать validate(hypergraph). При наличии errors не выполнять run, выбросить исключение (например ValidationError с списком ошибок).
2. **num_loop_steps:** если None — взять hypergraph.metadata.get("num_loop_steps", 1). Используется для шагов плана типа "cycle".
3. **Ключи inputs:** по get_input_spec() построить отображение ключ_внешний → (node_id, port_name). Заполнить буфер: для каждого (node_id, port_name) из spec положить inputs[key] в buffer[(node_id, port_name)]. Ключ = name если задан, иначе "node_id:port_name". Поддержать также ключ (node_id, port_name) в inputs для совместимости.
4. **Устройство и режим:** если device задан — для каждого узла вызвать node.to(device), если такой метод есть (узел-задача = Block+Node, у блока есть to). Если training — node.train(True), иначе node.eval() (если есть).
5. **План:** plan = build_plan(hypergraph).
6. **Буфер:** buffer: Dict[Tuple[str, str], Any] = {}; инициализировать из inputs как выше.
7. **Обход по плану:**
   - Для шага ("node", node_id): собрать входы узла из buffer (по get_edges_in(node_id) и по exposed_input для этого узла); если dry_run — не вызывать выполнение, подставить None для выходов; иначе вызвать **node.run(inputs)** — по контракту фазы 1 (узел-задача: run делегирует в forward того же объекта, Block-часть); записать выходы в buffer по исходящим рёбрам; вызвать callbacks с phase "before"/"after".
   - Для шага ("cycle", (rep, comp)): node_order = sorted(comp); callbacks "loop_start"; для iteration в range(num_loop_steps): для каждого node_id в node_order выполнить то же, что для "node"; callbacks "loop_end".
8. **Сбор outputs:** по get_output_spec() собрать из buffer значения в словарь result; ключ = name или "node_id:port_name".
9. Вернуть result.

**Сбор входов узла (gather_inputs):**

- Для каждого входящего ребра (source_node, source_port) → (node_id, target_port): взять buffer[(source_node, source_port)] и положить в inputs[target_port].
- Для портов узла, которые в exposed_inputs и (node_id, port_name) есть в buffer (заполнены при инициализации) — добавить в inputs[port_name].
- Опциональные порты без значения: не передавать или передать default из config блока (как в reference).

**Запись выходов (apply_outputs):**

- Для каждого (port_name, value) из outputs узла: записать buffer[(node_id, port_name)] = value. Все исходящие рёбра ведут в (target_node, target_port) — при выполнении приёмников значение будет прочитано из buffer[(node_id, port_name)] (один источник на порт в фазе 2) или агрегировано, если реализована агрегация.

**Тесты (test_executor.py):**

- Цепочка A → B → C с IdentityBlock/AddBlock: inputs на порт A, run, проверка outputs с порта C.
- Один узел: exposed_input и exposed_output на его портах; run(inputs) == ожидаемый outputs.
- Цикл из двух узлов (A ↔ B), num_loop_steps=2: проверка, что узлы вызваны дважды и результат согласован с ожиданием.
- dry_run=True: block.forward не вызывается (можно мок), в buffer пишутся None для выходов; выходы графа по spec есть (None).
- validate_before=True и невалидный граф: run выбрасывает исключение, выполнение не идёт.
- callbacks: вызваны с правильными node_id и phase ("before", "after", "loop_start", "loop_end").

---

## 11. Контракт run — сводка

- **Вход (inputs):** словарь. Ключи — name (если задан в expose_input) или "node_id:port_name". Значения — данные для соответствующих портов.
- **Выход (outputs):** словарь. Ключи — name или "node_id:port_name" по get_output_spec(). Значения — результаты с соответствующих портов после выполнения всего плана.
- **Опции:** training, num_loop_steps, device, callbacks, dry_run, validate_before. Остальные (seed, max_steps для agent_loop) — фаза 3.

---

## 12. Порядок реализации

1. **Edge** — edge.py, test_edge.py.
2. **Hypergraph** — structure.py (add_node, add_edge, expose, get_*, execution_version), test_structure.py.
3. **Validator** — validator.py, ValidationResult, test_validator.py.
4. **Planner** — planner.py (Tarjan, топологический порядок SCC, build_plan, кэш), test_planner.py.
5. **Executor** — executor.py (run, буфер внутри, gather_inputs, apply_outputs, цикл по плану), test_executor.py.
6. **Интеграция** — общий тест: собрать граф из 3 узлов (helpers), два ребра, expose_input на первый узел, expose_output на последний, run и проверить outputs.
7. **Пакет engine** — __init__.py с экспортом Edge, Hypergraph, Validator, build_plan, run.

---

## 13. Тесты — сводка

| Компонент | Файл | Ключевые сценарии |
|-----------|------|-------------------|
| Edge | test_edge.py | создание, равенство, хэш |
| Hypergraph | test_structure.py | add_node, add_edge, expose, get_edges_in/out, get_input_spec/get_output_spec, execution_version |
| Validator | test_validator.py | валидный граф; отсутствующий узел/порт; несовместимые типы; обязательный вход без ребра |
| Planner | test_planner.py | DAG цепочка; цикл из двух узлов; один узел; кэш |
| Executor | test_executor.py | цепочка A→B→C; один узел; цикл K=2; dry_run; validate_before; callbacks |
| Интеграция | test_executor.py или test_engine_integration.py | полный run от inputs до outputs с реальными блоками из tests/foundation/helpers |

Тестовые блоки: использовать IdentityBlock, AddBlock из tests/foundation/helpers.py (фаза 1). В tests/engine/helpers.py можно определить фабрики графов: linear_graph(nodes_list), cycle_graph(node_ids).

---

## 14. Приёмочные критерии и сценарий «фреймворк запустился»

- [ ] Edge создаётся и участвует в равенстве/хэше.
- [ ] Hypergraph: add_node, add_edge, expose_input, expose_output работают; get_node, get_edges_in/out, get_input_spec/get_output_spec возвращают ожидаемое; execution_version растёт при изменениях.
- [ ] Validator на валидном графе возвращает пустые errors; на невалидном — хотя бы одну ошибку.
- [ ] Planner для цепочки A→B→C даёт порядок [A, B, C]; для пары A↔B даёт один шаг "cycle".
- [ ] run(hypergraph, inputs) с цепочкой из трёх узлов (например IdentityBlock) и одним входом/одним выходом возвращает словарь outputs с ожидаемым ключом и значением.
- [ ] run с num_loop_steps=2 для графа с циклом из двух узлов выполняет внутренние узлы дважды.
- [ ] dry_run=True не вызывает block.forward; выходы по spec присутствуют (значения могут быть None).
- [ ] Все тесты tests/engine/ проходят (pytest tests/engine/ -v).

**Минимальный сценарий «запустился»:** в тесте или примере: создать Hypergraph; добавить три **узла-задачи** (объекты Block+Node, например заглушки backbone/identity или IdentityBackbone из фазы 4); add_edge(A.out, B.in), add_edge(B.out, C.in); expose_input(A, "in", "x"), expose_output(C, "out", "y"); run(h, {"x": value}) → {"y": value}. Гиперграф собирается только из узлов-задач. Это подтверждает, что движок и структура работают от входа до выхода.

---

## 15. Граничные случаи

| Случай | Ожидание |
|--------|----------|
| Пустой граф (0 узлов) | План пустой; run возвращает пустой dict (или ошибка валидации — «нет узлов»). |
| Один узел, без рёбер | План из одного шага ("node", id). Входы только из exposed_inputs; выходы только из exposed_outputs. |
| Дубликат add_edge | Идемпотентно: не добавлять второе такое же ребро; execution_version можно не менять при повторном add_edge. |
| remove_node | Удалить все рёбра, где узел участвует; убрать записи из exposed_inputs/outputs для этого node_id. |
| Непереданный обязательный внешний вход | Валидатор: порт в exposed_inputs — считаем, что вызывающий обязан передать. При run отсутствующий ключ в inputs — ошибка (или явно задокументировать: «все ключи из get_input_spec() должны быть в inputs для обязательных портов»). |
| Опциональный порт без входящего ребра | Допустимо; при gather_inputs не передавать ключ или передать default из блока. |
| num_loop_steps=0 для цикла | Циклическая фаза не выполняется (0 итераций); буферы для узлов внутри цикла не обновляются от этих узлов. |
| Callback бросает исключение | Не прерывать run (проглатывать, как в reference). |

---

## 16. Референс: outdated_1

- **reference/outdated_1/foundation/graph.py** — Graph с _nodes, _edges, _in_edges_by_node, _out_edges_by_node, _exposed_inputs, _exposed_outputs, _execution_version; add_node(node_id, block, ...), add_edge(Edge), expose_input, expose_output, get_input_spec, get_output_spec, _validate_edge при add_edge. Использовать как образец структуры и сигнатур; в фазе 2 узел — AbstractGraphNode (не block напрямую), add_node(node_id, node).
- **reference/outdated_1/executor/run.py** — _input_key, _output_key (name или "node_id:port_name"); _scc_tarjan, _topological_order_sccs, _build_execution_plan (план ["node"|"cycle"]); _gather_inputs_for_node (по get_edges_in и buffer, опциональные порты с defaults); _apply_outputs_to_buffer; run(graph, inputs, training, num_loop_steps, device, callbacks, dry_run). Логику перенести в executor и planner; граф заменить на Hypergraph с get_node, get_edges_in, get_input_spec, get_output_spec, metadata.

Отличия от референса в фазе 2:

- Класс структуры — Hypergraph (не Graph); узлы — AbstractGraphNode.
- Вызов узла: node.run(inputs) (контракт фазы 1); у узла-задачи run делегирует в forward того же объекта (Block-часть). Обёртки «узел с блоком» нет — в графе только объекты Block+Node.
- Edge — отдельный тип в engine/edge.py; в Graph референса Edge импортируется из foundation.

После выполнения фазы 2 фреймворк способен выполнять гиперграф задачи (цепочки и один цикл) без сериализации и без agent_loop; фаза 3 добавит from_config, реестр, сериализацию и при необходимости agent_loop.

---

## Итог

Фаза 2 задаёт **полную спецификацию гиперграфового движка**: Edge, Hypergraph (хранилище структуры с add_node, add_edge, expose, get_*), Validator, Planner (Tarjan SCC + топологический порядок, кэш), буферы (внутри executor), Executor и контракт **run(hypergraph, inputs, **options) → outputs**. Реализация в указанном порядке с тестами по каждому компоненту и интеграционным сценарием (цепочка из трёх узлов и цикл из двух) обеспечивает запуск фреймворка «с первого раза»: сборка графа и вызов run дают воспроизводимый результат. Документ опирается на канон (01, 03, HYPERGRAPH_ENGINE), план реализации и референс outdated_1 (graph.py, run.py).
