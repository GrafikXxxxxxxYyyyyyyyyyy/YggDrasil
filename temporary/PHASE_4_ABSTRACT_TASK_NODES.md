# Фаза 4. Абстрактные узлы-задачи (семь ролей) — полный технический план

Детальный технический план по **четвёртой фазе**: введение семи ролей узлов-задач (Backbone, Injector, Conjector, Inner Module, Outer Module, Helper, Converter) как абстрактных классов и их минимальных заглушек (stubs), регистрация в реестре типов блоков, опционально — автосвязывание по ролям и подготовка к agent_loop. Цель — чтобы гиперграф задачи можно было собирать из узлов по **block_type** (например `backbone/identity`, `conjector/identity`, `inner_module/identity`), валидировать и выполнять run; семь ролей задают контракты портов и типичные связи по канону. Документ опирается на канон (01, 02), план реализации, фазы 1–3 и референс (outdated_1/task_nodes).

**Связь с планом:** [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — Фаза 4.

**Канон:** [documentation/docs/02_ABSTRACT_TASK_NODES.md](../documentation/docs/02_ABSTRACT_TASK_NODES.md), [documentation/docs/01_FOUNDATION.md](../documentation/docs/01_FOUNDATION.md).

**Референс:** reference/outdated_1/task_nodes/ (abstract.py, stubs.py, roles.py, role_rules.py, auto_connect.py).

**Язык:** русский.

---

## 1. Цель фазы 4

Реализовать **уровень абстрактных узлов-задач** так, чтобы:

- **Семь ролей** были задекларированы как абстрактные классы (наследники AbstractBaseBlock) с фиксированными контрактами портов и семантикой по документу 02.
- Конкретные реализации (в т.ч. **заглушки** для тестов и сборки графа) регистрировались в реестре под типами вида `backbone/identity`, `conjector/identity`, `inner_module/identity` и т.д.
- Гиперграф задачи можно было собирать из конфига или программно, указывая **block_type** узла (например `backbone/identity`, `converter/identity`), и выполнять **run**; валидация и движок работают с портами, объявленными абстракциями.
- По **block_type** можно было однозначно определить **роль** (role_from_block_type) для автосвязывания и валидации шаблонов.
- Опционально: **автосвязывание** (auto_connect) при добавлении узла — предложение или автоматическое создание рёбер по правилам «типичных связей» между ролями (02 §12.2).

**Результат фазы 4:** семь ролей объявлены и доступны через реестр; минимальные заглушки позволяют собрать тестовый гиперграф (например цепочка Converter → Conjector → Backbone или цикл Backbone ↔ Inner Module) и прогнать run; при необходимости включено автосвязывание по ролям. Agent_loop (узел-агент, tool_calls) может быть заглушкой или отложен на следующую фазу.

---

## 2. Зависимости

- **Фаза 0:** структура репозитория, pytest. См. [PHASE_0_STRUCTURE.md](PHASE_0_STRUCTURE.md).
- **Фаза 1:** Port, PortDirection, PortType, PortAggregation, AbstractBaseBlock, AbstractGraphNode, BlockRegistry, register_block. Блок объявляет порты через declare_ports(), выполняет forward(inputs)→outputs, имеет block_type и block_id. См. [PHASE_1_FOUNDATION.md](PHASE_1_FOUNDATION.md).
- **Фаза 2:** Edge, Hypergraph, Validator, Planner, Executor, run(hypergraph, inputs). См. [PHASE_2_ENGINE.md](PHASE_2_ENGINE.md).
- **Фаза 3:** Hypergraph с add_node(node_id, block_type, config, ...), from_config, to_config, run, get_input_spec, get_output_spec. См. [PHASE_3_TASK_HYPERGRAPH.md](PHASE_3_TASK_HYPERGRAPH.md).

По канону 02 §3.1: **два равноправных начала** — AbstractBaseBlock (данные, вычисления, **материальная** сущность) и AbstractGraphNode (положение, связи в графе, **идеальное** начало). Их не смешивать и не подменять одно другим; оба одинаково необходимы. Узлы-задачи **наследуют оба** начала; только при двойном наследовании они могут быть встроены в граф и исполнены (что хранят и вычисляют — отвечает блок; как встроены в структуру — отвечает узел). Узлы-задачи олицетворяют **стремление быть реализованными** конкретной реализацией в гиперграфе. **Гиперграф задачи** собирается **только из узлов-задач** (они первые и единственные, кто наследует и Block, и Node) и является **цельной единицей системы**, отвечающей за **полную завершённую задачу**. Ниже в §4 и §7 — разделение ответственности и реализация абстракций.

---

## 3. Что входит в фазу 4 и что нет

| Входит в фазу 4 | Не входит (фазы 3, 5 и далее) |
|-----------------|--------------------------------|
| Семь абстрактных классов: AbstractBackbone, AbstractInjector, AbstractConjector, AbstractInnerModule, AbstractOuterModule, AbstractHelper, AbstractConverter. Каждый — наследник AbstractBaseBlock, declare_ports() по канону 02, абстрактный forward(). | Реальные реализации (UNet, DDIM, VAE, CLIP, LoRA и т.д.) — отдельные пакеты или фазы; в фазе 4 достаточно заглушек. |
| Константы ролей (backbone, injector, conjector, inner_module, outer_module, helper, converter) и функция role_from_block_type(block_type). | |
| Заглушки (stubs): по одной минимальной реализации на роль (identity/passthrough), регистрация как role/identity. | |
| Регистрация заглушек в глобальный реестр (при импорте пакета task_nodes или явном вызове register_all_stubs()). | Сериализация в файлы (фаза 5). |
| Опционально: правила типичных связей между ролями (role_rules), автосвязывание (suggest_edges, apply_auto_connect, use_task_node_auto_connect). | Agent_loop в движке (подцикл агент + инструменты) — можно заглушить или реализовать в фазе 3/4. |
| Тесты: сборка гиперграфа из конфига с block_type узлов-задач, run, проверка выходов; тесты каждой заглушки; role_from_block_type; при наличии — автосвязывание. | |

---

## 4. Двойное наследование: разделение ответственности

### 4.1 Разделение ролей

- **AbstractBaseBlock** — **материальное** начало: только данные и вычисления (forward, block_type, block_id, state_dict). **Не объявляет порты** — порты у узла. Не знает о гиперграфе и положении в нём.
- **AbstractGraphNode** — **идеальное** начало: положение и связи в гиперграфе (node_id, **объявление портов** declare_ports, get_input_ports, get_output_ports, run). Не хранит блок. Даёт движку интерфейс портов и run(inputs)→self.forward(inputs); у узла-задачи declare_ports и forward реализует один класс (два начала в одном объекте).

В 02 §3.1 задано: каждая абстрактная сущность узла-задачи **наследует** и AbstractBaseBlock, и AbstractGraphNode. Один класс, один объект — две ответственности в одном экземпляре. Обёртывания блока в узел нет. В коде:

- **Абстрактные классы ролей** объявляются как наследники обоих базовых классов, например:  
  `class AbstractBackbone(AbstractBaseBlock, AbstractGraphNode):`
- **Конкретная реализация** (в т.ч. заглушка) тоже наследует только абстракцию роли (которая уже наследует Block и Node), например:  
  `class IdentityBackbone(AbstractBackbone):`
- **Экземпляр** такого класса создаётся с `node_id` и параметрами блока (config, block_id); один объект имеет и интерфейс блока (declare_ports, forward, block_type), и интерфейс узла (node_id, run, get_input_ports, get_output_ports).
- **В гиперграфе** хранится именно этот объект: при add_node(node_id, block_type, config) реестр возвращает один объект «блок+узел»; node_id задаётся ему (в конфиге или при добавлении); в графе сохраняется соответствие node_id → этот объект. Отдельной обёртки AbstractGraphNode(block=...) для узлов-задач не используется.

### 4.2 Граф собирается только из узлов-задач

**Гиперграф задачи** составляется **только из абстрактных узлов-задач** (экземпляров классов, наследующих и Block, и Node). Отдельной сущности «узел с блоком внутри» (обёртка) не используется. add_node(node_id, block_type, config) вызывает реестр; реестр возвращает объект — узел-задачу (Block+Node в одном лице); этот объект сохраняется в графе по node_id. Движок обходит только такие объекты и вызывает у них run(inputs).

### 4.3 Реализация AbstractGraphNode (фаза 1)

AbstractGraphNode **не принимает блок в конструкторе**. Он предназначен **только для наследования** вместе с AbstractBaseBlock в классах узлов-задач. Конструктор: `__init__(self, node_id: str)`. Методы интерфейса для движка: get_input_ports() и get_output_ports() строятся из self.declare_ports() (метод Node-части того же объекта); run(inputs) возвращает self.forward(inputs). AbstractGraphNode отвечает только за node_id и за доступ к портам/выполнению для движка; данные и вычисления — в Block-части того же объекта.

### 4.4 Создание объекта и добавление в граф

- **registry.build(config)** для block_type узла-задачи возвращает один объект (экземпляр класса Block+Node). В config передаются node_id, block_id и параметры блока. Конструктор вызывает оба родительских __init__.
- **add_node(node_id, block_type, config)** в Hypergraph: build_config включает node_id; полученный объект кладётся в _nodes[node_id]. В графе хранятся только такие объекты (узлы-задачи).
- **Движок** вызывает node.run(inputs); run (из AbstractGraphNode) делегирует в self.forward(inputs) того же объекта.


---

## 5. Размещение кода (структура пакета)

```
yggdrasill/
  foundation/           # фаза 1
  engine/               # фазы 2–3
  task_nodes/           # новая папка фазы 4
    __init__.py         # экспорт абстракций, заглушек, ролей, register_all_stubs
    roles.py            # константы ролей (BACKBONE, INJECTOR, ...), role_from_block_type
    abstract.py         # AbstractBackbone, AbstractInjector, AbstractConjector,
                        # AbstractInnerModule, AbstractOuterModule, AbstractHelper, AbstractConverter
    stubs.py            # IdentityBackbone, IdentityInjector, ... (заглушки), @register_block
    role_rules.py       # (опционально) правила портов по парам ролей, suggest_edges_for_new_node
    auto_connect.py     # (опционально) apply_auto_connect, use_task_node_auto_connect

tests/
  task_nodes/
    __init__.py
    test_roles.py       # role_from_block_type, константы
    test_abstract.py    # порты каждой абстракции (declare_ports), имена и направление
    test_stubs.py       # каждая заглушка: forward с тестовыми входами, регистрация в реестре
    test_hypergraph_with_roles.py  # гиперграф из конфига с nodes block_type=backbone/identity и т.д., run
    test_auto_connect.py # (опционально) добавление узла с auto_connect, проверка появления рёбер
```

Импорт после фазы 4:

```python
from yggdrasill.task_nodes import (
    AbstractBackbone, AbstractInjector, AbstractConjector,
    AbstractInnerModule, AbstractOuterModule, AbstractHelper, AbstractConverter,
    register_all_stubs,
)
from yggdrasill.task_nodes.roles import BACKBONE, INNER_MODULE, role_from_block_type
# при необходимости
from yggdrasill.task_nodes.auto_connect import use_task_node_auto_connect
```

При первом импорте task_nodes.stubs или вызове register_all_stubs() заглушки регистрируются в BlockRegistry.global_registry(), чтобы Hypergraph.from_config и add_node(block_type=...) их находили.

---

## 6. Роли и роль из block_type

### 6.1 Константы ролей

Имена ролей совпадают с префиксами block_type в каноне 02. Рекомендуемые константы (в `roles.py`):

```python
BACKBONE = "backbone"
INJECTOR = "injector"
CONJECTOR = "conjector"
INNER_MODULE = "inner_module"
OUTER_MODULE = "outer_module"
HELPER = "helper"
CONVERTER = "converter"

KNOWN_ROLES: Set[str] = {
    BACKBONE, INJECTOR, CONJECTOR, INNER_MODULE,
    OUTER_MODULE, HELPER, CONVERTER,
}
```

### 6.2 role_from_block_type(block_type: str) -> Optional[str]

Возвращает роль по строке block_type: если block_type равен константе роли или начинается с неё и слеша/подчёркивания (например `backbone`, `backbone/identity`, `backbone_unet2d`), вернуть соответствующую роль; иначе None. Используется автосвязыванием и при необходимости валидатором шаблонов. Реализация: перебор KNOWN_ROLES, проверка `bt == role` или `bt.startswith(role + "/")` или `bt.startswith(role + "_")` (после strip().lower() для block_type).

Тесты: role_from_block_type("backbone") == "backbone"; role_from_block_type("backbone/identity") == "backbone"; role_from_block_type("unknown") is None; для каждой роли подтип вида role/identity возвращает эту роль.

---

## 7. Абстрактные классы по ролям

Каждый абстрактный класс: (1) наследует AbstractBaseBlock; (2) реализует declare_ports() — список Port по канону 02 для этой роли; (3) задаёт block_type как свойство (возвращает константу роли или подтип в конкретной реализации); (4) объявляет абстрактный forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]. Конструктор и config — по контракту AbstractBaseBlock (block_id, config и т.д.). Ниже — порты по документу 02; типы портов (PortType) выбираются по смыслу (TENSOR, ANY, TEXT, IMAGE и т.д.).

### 7.1 AbstractBackbone

- **Назначение:** ядро задачи; один шаг предсказания (02 §4).
- **Порты (declare_ports):**
  - latent (IN, TENSOR) — текущее латентное состояние;
  - timestep (IN, TENSOR) — текущий шаг времени;
  - condition (IN, ANY, optional=True) — условие генерации;
  - pred (OUT, TENSOR) — предсказание модели.
- **block_type:** свойство, по умолчанию возвращает BACKBONE; в конкретном классе может возвращать "backbone/identity", "backbone/unet2d" и т.д.

### 7.2 AbstractInjector

- **Назначение:** встраивается внутрь ядра, влияет на генерацию изнутри (02 §5, например LoRA).
- **Порты:**
  - condition (IN, ANY) — сигнал для инжекции;
  - hidden (IN, TENSOR, optional=True) — промежуточные активации ядра при необходимости;
  - adapted (OUT, TENSOR) — модифицированные активации или дельта.
- **block_type:** INJECTOR или подтип.

### 7.3 AbstractConjector

- **Назначение:** стоит рядом с ядром, поставляет условие (02 §6, например CLIP-энкодер).
- **Порты:**
  - input (IN, ANY) — вход для формирования условия (prompt, context);
  - condition (OUT, TENSOR или ANY) — условие в Backbone.
- **block_type:** CONJECTOR или подтип.

### 7.4 AbstractInnerModule

- **Назначение:** встраивается внутрь цикла; один шаг перехода или вклад в итерацию (02 §7, например солвер DDIM).
- **Порты (минимум для солвера):**
  - latent (IN, TENSOR);
  - timestep (IN, TENSOR);
  - pred (IN, TENSOR);
  - control (IN, TENSOR или ANY, optional=True);
  - next_latent (OUT, TENSOR);
  - next_timestep (OUT, TENSOR, optional=True).
- Для control-модулей внутри итерации можно добавить порт output (OUT, TENSOR, optional). В базовой абстракции достаточно перечисленных шести портов.
- **block_type:** INNER_MODULE или подтип.

### 7.5 AbstractOuterModule

- **Назначение:** рядом с циклом: до входа в цикл или после выхода (02 §8, например начальный шум, расписание).
- **Порты:**
  - input (IN, TENSOR или ANY, optional=True) — результат цикла, если модуль обрабатывает выход цикла;
  - output (OUT, TENSOR или ANY) — вход в цикл или сигнал снаружи.
- Модуль может быть только источником (только output, без входящих рёбер) или только приёмником выхода цикла (input → output). В declare_ports объявить оба порта; input — optional.
- **block_type:** OUTER_MODULE или подтип.

### 7.6 AbstractHelper

- **Назначение:** вспомогательная функция, не покрытая остальными шестью ролями (02 §10, RAG, файлы, API).
- **Порты:**
  - query (IN, ANY) — запрос;
  - result (OUT, ANY) — результат.
- **block_type:** HELPER или подтип.

### 7.7 AbstractConverter

- **Назначение:** преобразование данных (формат, кодировка) (02 §9, VAE, токенизатор).
- **Порты:**
  - input (IN, ANY);
  - output (OUT, ANY).
- Конкретные реализации могут расширять (несколько пар вход/выход, например encode_image, encode_latent, decode_latent, decode_image); минимальная абстракция — input/output.
- **block_type:** CONVERTER или подтип.

### 7.8 Общие требования к абстракциям

- Все классы в abstract.py; импорт Port, PortDirection, PortType из foundation.port; AbstractBaseBlock из foundation.block.
- В каждой абстракции: declare_ports() возвращает список Port с указанными именами и направлениями; не возвращать None; имена портов уникальны внутри блока.
- Абстрактный метод forward(inputs) -> outputs; в реализациях (заглушках) возвращать словарь с ключами по именам выходных портов.

---

## 8. Заглушки (stubs)

Для каждой роли — один класс-заглушка: минимальная реализация forward, передающая входы на выходы (identity/passthrough) или возвращающая фиксированные значения, чтобы граф мог выполниться без реальных моделей.

### 8.1 Регистрация

Каждая заглушка регистрируется в реестре под типом **role/identity** (например backbone/identity, conjector/identity). Использовать декоратор @register_block("backbone/identity") или явный вызов registry.register("backbone/identity", IdentityBackbone) при загрузке модуля. Функция **register_all_stubs(registry=None)** регистрирует все заглушки в переданном реестре или в BlockRegistry.global_registry().

### 8.2 Поведение заглушек

- **IdentityBackbone:** pred = inputs.get("latent") (или нулевой тензор по конфигу, если latent нет).
- **IdentityInjector:** adapted = inputs.get("condition").
- **IdentityConjector:** condition = inputs.get("input").
- **IdentityInnerModule:** next_latent = inputs.get("latent"), next_timestep = inputs.get("timestep"); при отсутствии портов — по умолчанию передать что есть.
- **IdentityOuterModule:** output = inputs.get("input") если input есть, иначе константа/пустой тензор по конфигу (или один выход с дефолтом).
- **IdentityHelper:** result = inputs.get("query").
- **IdentityConverter:** output = inputs.get("input").

Заглушки должны соблюдать контракт портов: возвращать словарь только с ключами — именами выходных портов, объявленных в соответствующей абстракции; для опциональных выходов можно не включать ключ, если движок это допускает.

### 8.3 block_type в заглушках

В каждом классе-заглушке переопределить block_type так, чтобы возвращался подтип "role/identity" (например "backbone/identity"). role_from_block_type("backbone/identity") должен возвращать "backbone".

---

## 9. Правила типичных связей и автосвязывание (опционально)

По канону 02 §12.2 типичные связи между ролями заданы в подразделах «Типичные связи» для каждой роли. Для автосвязывания при добавлении нового узла нужно:

- **role_rules:** таблица (source_role, target_role) → [(source_port, target_port), ...]. Примеры: (conjector, backbone) → [(condition, condition)]; (backbone, inner_module) → [(pred, pred)]; (inner_module, backbone) → [(next_latent, latent), (next_timestep, timestep)]; (converter, conjector) → [(output, input)]; (outer_module, backbone) → [(output, latent)] и т.д. Полный набор — по 02 §§4.5–10.5.
- **suggest_edges_for_new_node(new_node_id, new_role, existing_roles_by_node_id)** — возвращает список (source_node_id, source_port, target_node_id, target_port) для рёбер, которые можно добавить: для каждой пары (existing_role, new_role) и (new_role, existing_role) применить правила и сформировать концы рёбер с учётом node_id.
- **apply_auto_connect(hypergraph, new_node_id, new_block)** — для нового узла вычислить new_role = role_from_block_type(block_type); собрать existing_roles_by_node_id по остальным узлам графа; вызвать suggest_edges_for_new_node; для каждого предложенного ребра проверить существование портов и совместимость типов (get_port, compatible_with); если ребро ещё не существует, вызвать hypergraph.add_edge(Edge(...)).
- **use_task_node_auto_connect(hypergraph)** — установить у гиперграфа callback (например auto_connect_fn = apply_auto_connect), чтобы при add_node(..., auto_connect=True) вызывался этот callback. Гиперграф в фазе 3 может принимать параметр auto_connect и вызывать callback после добавления узла (см. PHASE_3).

Если в фазе 4 автосвязывание не реализуется, оставить заглушку (apply_auto_connect — no-op или только логирование) и описать в документе формат role_rules и suggest_edges_for_new_node для последующей реализации.

---

## 10. Agent_loop (опционально)

По канону 03 §7.3 и HYPERGRAPH_ENGINE движок поддерживает agent_loop: узел-агент в выходе может вернуть tool_calls; движок выполняет узлы-инструменты по таблице tool_id → node_id и передаёт tool_results агенту, повторяя вызов до отсутствия tool_calls или до max_steps. В фазе 4 можно:

- Не реализовывать agent_loop в движке (оставить на фазу 3 или отдельную задачу).
- Ввести абстрактный узел-агент как подтип Backbone или отдельную роль (например block_type agent/...) с контрактом: выход может содержать поле tool_calls; конфиг гиперграфа задаёт mapping tool_id → node_id. Заглушка агента без tool_calls просто возвращает response.

Если реализуется: в executor при выполнении узла проверять выход на наличие tool_calls; при наличии — выполнить узлы-инструменты, собрать tool_results, снова вызвать узел с дополненными входами; лимит max_steps. Детали — по HYPERGRAPH_ENGINE и 03 §7.3.

---

## 11. Порядок реализации

1. **roles.py** — константы ролей, KNOWN_ROLES, role_from_block_type; тесты test_roles.py.
2. **abstract.py** — семь абстрактных классов с declare_ports() и абстрактным forward(); тесты test_abstract.py (проверка имён и направлений портов для каждого класса).
3. **stubs.py** — семь заглушек, @register_block("role/identity"), register_all_stubs(); тесты test_stubs.py (создание экземпляра через реестр, вызов forward с тестовыми inputs, проверка ключей в outputs).
4. **Интеграция с гиперграфом:** тест test_hypergraph_with_roles.py — конфиг с nodes с block_type backbone/identity, inner_module/identity, converter/identity и т.д., edges и exposed_inputs/exposed_outputs; Hypergraph.from_config(config, registry); hypergraph.run(inputs); проверка outputs. Убедиться, что register_all_stubs() вызван (например при импорте stubs или в conftest).
5. **role_rules.py** (опционально) — таблица правил, get_rule_edges, suggest_edges_for_new_node; тесты test_role_rules.py.
6. **auto_connect.py** (опционально) — apply_auto_connect, use_task_node_auto_connect; тесты test_auto_connect.py (добавление узла с auto_connect, проверка появления ожидаемых рёбер).
7. **Пакет task_nodes** — __init__.py с экспортом абстракций, заглушек, ролей, register_all_stubs; при импорте пакета по желанию вызывать register_all_stubs() для глобального реестра.

---

## 12. Тесты — сводка

| Что тестировать | Где | Ожидание |
|-----------------|-----|----------|
| role_from_block_type | test_roles.py | Для каждой роли и подтипа role/identity возвращается правильная роль; для неизвестного типа None. |
| declare_ports каждой абстракции | test_abstract.py | Количество портов, имена, direction IN/OUT, совпадение с каноном 02. |
| Заглушка каждой роли | test_stubs.py | registry.build({"block_type": "backbone/identity"}) возвращает экземпляр; forward( inputs ) возвращает dict с ожидаемыми ключами (имена выходных портов). |
| Гиперграф из конфига с узлами-задачами | test_hypergraph_with_roles.py | from_config с nodes block_type backbone/identity, inner_module/identity, converter/identity; add_edge по портам; run(inputs) возвращает outputs по exposed_outputs; структура графа (цепочка или цикл) выполняется без ошибок. |
| Автосвязывание (если есть) | test_auto_connect.py | Добавить узел с auto_connect=True после use_task_node_auto_connect; проверить, что появились рёбра по правилам для данной роли. |

---

## 13. Приёмочные критерии и граничные случаи

### 13.1 Приёмочные критерии

- [ ] Семь абстрактных классов объявлены, у каждого declare_ports() возвращает порты по канону 02.
- [ ] Семь заглушек зарегистрированы под типами role/identity; registry.build({"block_type": "backbone/identity"}) и аналоги для остальных ролей создают экземпляры.
- [ ] role_from_block_type(block_type) возвращает роль для всех подтипов вида role/identity и для префикса роли.
- [ ] Гиперграф, собранный из конфига с узлами block_type backbone/identity, inner_module/identity, converter/identity (и при необходимости outer_module/identity, conjector/identity), с заданными рёбрами и exposed_inputs/exposed_outputs, успешно выполняется: run(inputs) возвращает словарь outputs без исключений; при необходимости проверяется значение/форма выходов.
- [ ] Все тесты в tests/task_nodes/ проходят.

### 13.2 Граничные случаи

| Случай | Ожидание |
|--------|----------|
| Конкретный блок реализует не все канонические порты роли | Допустимо: абстракция задаёт типичный контракт; реализация может объявлять подмножество или расширение (02). Валидатор графа проверяет только существующие порты при добавлении рёбер. |
| Регистрация двух классов под одним block_type | Поведение реестра: последний зарегистрированный выигрывает или ошибка — по контракту Phase 1. |
| Вызов forward заглушки без обязательного входа | По контракту блока: optional порты могут отсутствовать; обязательные — либо значение по умолчанию в заглушке, либо KeyError/поведение по реализации. Заглушки предпочтительно обрабатывают отсутствующие ключи (get с default). |
| Автосвязывание при отсутствии совместимых портов у существующих узлов | Не добавлять ребро; не падать (пропуск или логирование). |

---

## 14. Референс: outdated_1 (task_nodes)

- **abstract.py** — абстрактные классы по ролям (в референсе: Backbone, Solver, Codec, Conditioner, Tokenizer, Adapter, Guidance). У нас семь ролей по 02: Backbone, Injector, Conjector, Inner Module, Outer Module, Helper, Converter. Соответствие: Solver → Inner Module, Conditioner → Conjector, Codec/Tokenizer → Converter, Adapter → Injector; Guidance в референсе — отдельная роль (CFG), у нас может быть часть Conjector или Helper; Outer Module и Helper в референсе явно не выделены. Порты и имена брать из 02.
- **stubs.py** — Identity-реализации для каждой роли, @register_block("role/identity"), register_task_node_stubs(registry). Использовать как образец регистрации и минимального forward.
- **roles.py** — константы ролей, role_from_block_type(block_type). Адаптировать под наши семь ролей.
- **role_rules.py** — таблица (source_role, target_role) → [(src_port, tgt_port)], suggest_edges_for_new_node. Адаптировать правила под порты из 02 (типичные связи §§4.5–10.5).
- **auto_connect.py** — apply_auto_connect(graph, new_node_id, new_block), use_task_node_auto_connect(graph). Граф вызывает callback после add_node при auto_connect=True. Гиперграф в нашем движке — Hypergraph из engine; при необходимости добавить атрибут auto_connect_fn и вызов в add_node (Phase 3).

Отличия от референса: семь ролей по канону 02; имена портов и типы строго по 02; блоки не наследуют Node — узел гиперграфа остаётся AbstractGraphNode с полем block.

---

## Итог

Фаза 4 задаёт **полный технический план уровня абстрактных узлов-задач**: **двойное наследование** (канон 02 §3.1) — Block только за данные и вычисления, Node только за положение и связи; граф собирается **только из узлов-задач** (один объект = Block+Node), без обёртки блока в узел. Семь ролей как абстрактные классы с контрактами портов по канону 02, заглушки role/identity и их регистрация в реестре, роль из block_type (role_from_block_type), опционально правила связей и автосвязывание, опционально agent_loop. Документ опирается на канон 01–02 и референс outdated_1/task_nodes.
