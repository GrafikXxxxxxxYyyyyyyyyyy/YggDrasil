# Фаза 1. Фундамент: Block, AbstractGraphNode, Port, реестр — полный технический план

Детальный технический план по **первой фазе** реализации: порты (Port), абстрактный блок (Abstract Base Block), узел гиперграфа (Abstract Graph Node), реестр типов блоков (Registry). Что реализовать, как это должно выглядеть (API, сигнатуры, контракты), как тестировать и как проверять. Канон — единственный источник истины.

**Связь с планом:** [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — Фаза 1.

**Канон:** [documentation/docs/01_FOUNDATION.md](../documentation/docs/01_FOUNDATION.md) — §§ 2 (Block), 2.4 (Port), 3 (Abstract Graph Node), 7 (Registry), 2.7 (сериализация блока).

**Язык:** русский.

---

## 1. Цель фазы 1

Реализовать **минимальный фундамент**, на котором строятся все остальные уровни:

- **Port** — описание одного входа или выхода блока (имя, тип, опциональность, политика агрегации).
- **Abstract Base Block** — абстрактный блок: **материальная** сущность — идентичность (block_type, block_id), хранение данных, выполнение forward(inputs)→outputs, сериализация state_dict/load_state_dict. Не объявляет порты (порты у узла). Отвечает только за данные и вычисления; не знает о графе и положении в нём.
- **Abstract Graph Node** — узел гиперграфа: **идеальное** начало — положение и связи (node_id, объявление портов declare_ports, get_input_ports, get_output_ports, run). Не хранит блок. Используется только в двойном наследовании с Block (узлы-задачи); run делегирует в self.forward того же объекта.
- **Registry** — реестр типов блоков: регистрация block_type → класс/фабрика, создание экземпляра по конфигу (build(config)).

**Результат фазы 1:** можно создавать блоки по типу из реестра, вызывать block.forward(inputs); определены два равноправных начала — Block (данные, вычисление, без портов) и Node (положение, связи, объявление портов); узлы-задачи (наследующие оба) будут встраиваться в гиперграф на фазах 2–4. Все тесты в `tests/foundation/` проходят.

---

## 2. Размещение кода и зависимость от фазы 0

- Вся реализация фазы 1 лежит в пакете **yggdrasill.foundation**.
- Фаза 0 должна быть выполнена: каталоги `yggdrasill/`, `yggdrasill/foundation/`, `tests/`, `tests/foundation/` существуют; пакет устанавливается через `pip install -e .`; тесты запускаются через `pytest tests/`.

**Целевая структура файлов после фазы 1:**

```
yggdrasill/foundation/
├── __init__.py    # Экспорт: Port, AbstractBaseBlock, AbstractGraphNode, Registry, типы портов и т.д.
├── port.py        # Port, PortDirection, PortType, PortAggregation
├── block.py       # AbstractBaseBlock
├── node.py        # AbstractGraphNode (узел гиперграфа)
└── registry.py    # BlockRegistry, register_block (декоратор)
```

Импорт из пакета после фазы 1:

```python
from yggdrasill.foundation import Port, AbstractBaseBlock, AbstractGraphNode, BlockRegistry
# или
from yggdrasill.foundation.port import Port, PortDirection, PortType, PortAggregation
from yggdrasill.foundation.block import AbstractBaseBlock
from yggdrasill.foundation.node import AbstractGraphNode
from yggdrasill.foundation.registry import BlockRegistry, register_block
```

---

## 3. Port (порт)

### 3.1 Назначение по канону

Порт задаёт **один вход или один выход** блока. У порта: имя (уникальное внутри блока), тип или схема данных, опциональность (обязательный/опциональный), при необходимости — политика агрегации для входа с несколькими входящими рёбрами. Совместимость при соединении узлов проверяется по типам портов (источник — выход, приёмник — вход; типы совместимы). Канон: 01 §2.3.1, §2.4.

### 3.2 Атрибуты и типы

| Атрибут | Тип | Обязательность | Описание |
|---------|-----|----------------|----------|
| **name** | str | да | Имя порта; уникально внутри блока. Не пустая строка. |
| **direction** | PortDirection (enum) | да | IN или OUT. Входной порт получает данные; выходной отдаёт. |
| **dtype** | PortType (enum) или str | нет (по умолчанию ANY) | Тип данных: TENSOR, DICT, ANY, IMAGE, TEXT, AUDIO, VIDEO или произвольная строка для расширения. Используется при проверке совместимости соединений. |
| **optional** | bool | нет (по умолчанию False) | Только для входных портов. Если True, блок должен корректно работать при отсутствии значения (значение по умолчанию или пропуск). |
| **aggregation** | PortAggregation (enum) | нет (по умолчанию SINGLE) | Только для входных портов при нескольких входящих рёбрах: SINGLE (ровно одно ребро), CONCAT, SUM, FIRST, DICT и т.д. Для выходного порта допустимо только SINGLE. |

**Перечисления (рекомендуемая реализация):**

- **PortDirection:** IN, OUT.
- **PortType:** TENSOR, DICT, ANY, IMAGE, TEXT, AUDIO, VIDEO (и при необходимости расширяемый набор).
- **PortAggregation:** SINGLE, CONCAT, SUM, FIRST, DICT (как минимум SINGLE, CONCAT, FIRST для фазы 1).

### 3.3 Контракт Port

- **Неизменяемость:** экземпляр Port после создания не должен изменяться (frozen dataclass или только чтение полей).
- **Валидация при создании:** name не пустой; для direction=OUT значение aggregation должно быть только SINGLE (иначе ValueError).
- **Метод совместимости:** `compatible_with(other: Port) -> bool`: True тогда и только тогда, когда self — выход (OUT), other — вход (IN), и типы совместимы (ANY совместим с любым; иначе совпадение dtype или правила подтипизации по соглашению).
- **Удобные свойства:** `is_input`, `is_output` — производные от direction.

### 3.4 Сигнатуры (псевдокод)

```python
class PortDirection(Enum):
    IN = "in"
    OUT = "out"

class PortType(Enum):
    TENSOR = "tensor"
    DICT = "dict"
    ANY = "any"
    # ...

class PortAggregation(Enum):
    SINGLE = "single"
    CONCAT = "concat"
    SUM = "sum"
    FIRST = "first"
    DICT = "dict"

@dataclass(frozen=True)
class Port:
    name: str
    direction: PortDirection
    dtype: PortType = PortType.ANY
    optional: bool = False
    aggregation: PortAggregation = PortAggregation.SINGLE

    @property
    def is_input(self) -> bool: ...

    @property
    def is_output(self) -> bool: ...

    def compatible_with(self, other: Port) -> bool: ...
```

### 3.5 Тесты для Port

Расположение: `tests/foundation/test_port.py`.

| Тест | Что проверяет |
|------|-------------------------------|
| Создание порта с name, direction, dtype | Все поля доступны и соответствуют переданным. |
| Пустое или пробельное name | ValueError при создании. |
| Выходной порт с aggregation != SINGLE | ValueError при создании. |
| is_input / is_output | True/False в зависимости от direction. |
| compatible_with: OUT → IN, одинаковый тип | True. |
| compatible_with: OUT(ANY) → IN(TENSOR) | True (ANY совместим с любым). |
| compatible_with: IN → OUT или OUT → OUT | False. |
| Несколько портов с разными aggregation | Создаются без ошибок; для IN допустимы CONCAT, FIRST. |

**Критерий прохождения:** все тесты в `test_port.py` зелёные; нет побочных эффектов при повторном вызове.

---

## 4. Abstract Base Block (блок)

### 4.1 Назначение по канону

Блок — **минимальная единица**: хранит (параметры, конфиг, веса, состояние) и вычисляет forward(inputs)→outputs. Контракт: идентичность (block_type, block_id), forward, сериализация (state_dict/load_state_dict). Порты не объявляет — это ответственность узла. Канон: 01 §2.

### 4.2 Идентичность

| Атрибут | Тип | Описание |
|---------|-----|----------|
| **block_type** | str | Тип блока для реестра и конфига (например "backbone/unet", "identity"). Определяет класс/реализацию. В подклассе переопределяется или задаётся через конфиг. |
| **block_id** | str | Уникальный в рамках графа (или области видимости) идентификатор экземпляра. Задаётся при создании или генерируется (например class_name + id(self)). |
| **config** | dict (read-only копия) | Параметры, переданные при создании; для воспроизводимости и сериализации. Не должен изменяться снаружи после создания. |

Правило: block_id при создании может быть None — тогда реализация генерирует уникальный id (например, по имени класса и id(self), или uuid).

### 4.3 Порты

- Блок **не объявляет порты**: порты относятся к **узлу** (идеальное начало, связи в графе). Блок только выполняет **forward(inputs: Dict) -> Dict** по контракту имён ключей; объявление портов (declare_ports, get_input_ports, get_output_ports) — ответственность AbstractGraphNode. В узле-задаче класс реализует и forward (Block), и declare_ports (Node).

### 4.4 Выполнение (run / forward)

- **forward(inputs: Dict[str, Any]) -> Dict[str, Any]** — абстрактный метод: по словарю входов возвращает словарь выходов. Имена ключей задаются контрактом реализации (в узле-задаче тот же класс реализует declare_ports на стороне Node — порты и имена ключей forward согласованы).

Блок не имеет метода run: вызов run(inputs) у графа выполняет узел (Node.run → self.forward того же объекта).

### 4.5 Сериализация состояния

- **state_dict() -> Dict[str, Any]** — возвращает словарь состояния блока (веса, кэш, всё необходимое для воспроизведения). Для блока без обучаемых параметров может возвращать пустой словарь или только конфигурируемые скаляры. Должен быть идемпотентным по смыслу (повторный вызов даёт согласованное состояние).
- **load_state_dict(state: Dict[str, Any], strict: bool = True) -> None** — восстанавливает состояние из словаря. strict=True: при наличии в state неожиданных ключей или отсутствии ожидаемых — выбросить исключение (KeyError или аналог). strict=False: загрузить то, что есть; лишние ключи игнорировать.

Для блоков с подблоками (композиция): state_dict может включать состояние подблоков с префиксом имени (например "child.offset"); load_state_dict разбирает префиксы и передаёт подсловари в подблоки. На фазе 1 достаточно реализовать базовое поведение для одного уровня; get_sub_blocks() можно вернуть пустой dict.

### 4.6 Опциональные аспекты (реализовать в базовом классе для единообразия)

- **Режим train/eval:** атрибут training (bool), методы train(mode=True), eval(). По умолчанию training=True. Используется в последующих фазах; на фазе 1 достаточно хранения флага.
- **Заморозка:** атрибут frozen (bool), методы freeze(), unfreeze(). По умолчанию frozen=False. Для фазы 1 достаточно флага; влияние на параметры — в фазах обучения.
- **Подблоки:** метод get_sub_blocks() -> Dict[str, AbstractBaseBlock]. По умолчанию возвращает {}. Если переопределён, state_dict/load_state_dict базового класса могут агрегировать/разбирать состояние подблоков по префиксам (по соглашению реализации).

### 4.7 Жизненный цикл (канон §2.5)

1. Создание: конструктор принимает block_id (опционально), config (опционально). Внутри присваиваются block_id (или сгенерированный), config (копия).
2. Размещение в графе: блок участвует в графе **только как часть узла-задачи** (двойное наследование Block+Node). В гиперграф добавляется объект узла-задачи (один объект = Block+Node), а не отдельный блок и не обёртка «узел с блоком».
3. Выполнение: движок вызывает node.run(inputs) у узла-задачи; run делегирует в forward того же объекта (Block-часть).
4. Сохранение/загрузка: state_dict() сохраняется; load_state_dict() восстанавливает состояние.

### 4.8 Сигнатуры (псевдокод)

```python
class AbstractBaseBlock(ABC):
    def __init__(self, block_id: Optional[str] = None, *, config: Optional[Dict[str, Any]] = None) -> None: ...

    @property
    def block_type(self) -> str: ...  # абстрактно или дефолт: type(self).__name__

    @property
    def block_id(self) -> str: ...

    @property
    def config(self) -> Dict[str, Any]: ...  # копия

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...

    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None: ...

    def get_sub_blocks(self) -> Dict[str, "AbstractBaseBlock"]: ...  # default: {}

    # опционально
    @property
    def training(self) -> bool: ...
    def train(self, mode: bool = True) -> "AbstractBaseBlock": ...
    def eval(self) -> "AbstractBaseBlock": ...
    @property
    def frozen(self) -> bool: ...
    def freeze(self) -> "AbstractBaseBlock": ...
    def unfreeze(self) -> "AbstractBaseBlock": ...
```
Порты (declare_ports, get_input_ports, get_output_ports) в Block **нет** — они у Node.

### 4.9 Тесты для Block

Расположение: `tests/foundation/test_block.py`. Нужны **конкретные блоки-заглушки** в `tests/foundation/helpers.py`: AddBlock (forward: входы "a", "b" → выход "out"; offset из config), IdentityBlock (forward: "x" → "y"). У блоков **нет** declare_ports — порты у узла.

| Тест | Что проверяет |
|------|-------------------------------|
| Создание блока с block_id и config | block_id, config, block_type доступны. |
| forward с полным словарём входов | Выход совпадает с ожидаемым (например a+b для AddBlock, x→y для IdentityBlock). |
| forward с отсутствующим ключом | По контракту реализации — KeyError или значение по умолчанию. |
| state_dict возвращает словарь | Для блока с состоянием (например offset) ключи и значения корректны. |
| load_state_dict затем forward | Поведение после загрузки воспроизводит сохранённое состояние. |
| load_state_dict strict=True с лишним ключом | Исключение. |
| train/eval, freeze/unfreeze | Флаги меняются; вызов eval() переводит training в False. |
| Блок с get_sub_blocks: state_dict содержит префиксы | Ключи вида "child.offset"; после load_state_dict подблок восстановлен. |

**Критерий прохождения:** все тесты в test_block.py зелёные; заглушки AddBlock и IdentityBlock используются и в test_registry. Порты и run тестируются на узле-задаче (test_node).

---

## 5. AbstractGraphNode (узел гиперграфа)

### 5.1 Назначение по канону

**Abstract Graph Node** (в коде — класс **AbstractGraphNode**) отвечает **только за положение и связи** в гиперграфе: node_id, **объявление портов** (declare_ports) и интерфейс для движка (get_input_ports, get_output_ports, run). Это **идеальное** начало (структура, связи, точки подключения в графе). **Материальное** начало — Block (данные, вычисление forward). Порты принадлежат узлу, не блоку. AbstractGraphNode не хранит блок; при двойном наследовании Node даёт node_id и declare_ports, Block даёт forward; run(inputs) делегирует в self.forward(inputs). Канон: 01 §3, 02 §3.1.

### 5.2 Атрибуты и контракт

| Атрибут | Тип | Описание |
|---------|-----|----------|
| **node_id** | str | Уникальный в рамках гиперграфа идентификатор узла. Не пустая строка. |

Гиперрёбра задаются на уровне графа. Узел предоставляет движку node_id и методы get_input_ports(), get_output_ports(), run(); при двойном наследовании они обращаются к self (Block-часть того же объекта).

### 5.3 Методы для движка

- **declare_ports()** — абстрактный метод узла: возвращает список Port (входы/выходы этой вершины в графе). Реализуется в классе узла-задачи; контракт имён согласован с forward (Block-часть того же класса).
- **get_input_ports()**, **get_output_ports()** — строятся из self.declare_ports() (фильтрация по direction IN/OUT). Нужны для построения графа и валидации рёбер.
- **run(inputs: Dict[str, Any]) -> Dict[str, Any]** — return self.forward(inputs) (forward из Block-части того же объекта).

Валидация при создании: node_id не пустой (иначе ValueError). AbstractGraphNode **не хранит блок**; порты объявляет узел (declare_ports), вычисление — блок (forward). См. [PHASE_4_ABSTRACT_TASK_NODES.md](PHASE_4_ABSTRACT_TASK_NODES.md) §4.

### 5.4 Сигнатуры (псевдокод)

```python
class AbstractGraphNode:
    def __init__(self, node_id: str) -> None: ...

    @property
    def node_id(self) -> str: ...

    @abstractmethod
    def declare_ports(self) -> List[Port]: ...  # объявление портов — ответственность узла

    def get_input_ports(self) -> List[Port]: ...
    def get_output_ports(self) -> List[Port]: ...
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...  # → self.forward(inputs)
```

### 5.5 Тесты для AbstractGraphNode

Расположение: `tests/foundation/test_node.py`.

| Тест | Что проверяет |
|------|-------------------------------|
| Создание узла-задачи (класс, наследующий Block и Node) с node_id | node_id доступен; get_input_ports/get_output_ports совпадают с declare_ports(); run(inputs) возвращает forward(inputs). |
| node_id пустая или пробелы | ValueError при создании. |
| node.run(inputs) == node.forward(inputs) | Делегирование в Block-часть того же объекта. |
| repr(node) или str | Содержит node_id и информацию о блоке (для отладки). |

**Критерий прохождения:** все тесты в test_node.py зелёные.

---

## 6. BlockRegistry (реестр типов блоков)

### 6.1 Назначение по канону

Реестр связывает **block_type** (строка) с классом или фабрикой. Позволяет создавать блоки по конфигу без жёсткого кода: в конфиге указан block_type и параметры; build(config) возвращает экземпляр блока. Новый тип блока добавляется регистрацией, без изменений ядра. Канон: 01 §7.

### 6.2 Операции

| Операция | Сигнатура | Описание |
|----------|-----------|----------|
| **register** | register(block_type: str, factory: Type[AbstractBaseBlock] \| Callable[..., AbstractBaseBlock]) -> None | Зарегистрировать тип. block_type — не пустая строка; factory — класс или фабричная функция, возвращающая блок. Повторная регистрация того же block_type перезаписывает (по соглашению). |
| **build** | build(config: Dict[str, Any]) -> AbstractBaseBlock | Создать экземпляр. В config обязательно поле "block_type" или "type"; остальные поля передаются в конструктор (в т.ч. block_id, config). Если block_type не зарегистрирован — KeyError с перечислением зарегистрированных. |
| **get** | get(block_type: str) -> Optional[...] | Получить зарегистрированную фабрику по типу; для интроспекции. |
| **__contains__** | block_type in registry -> bool | Проверка наличия типа. |

Правила build(config):

- config["block_type"] или config["type"] — ключ для поиска в реестре.
- В конструктор блока передаётся: block_id = config.get("block_id"), config = {k: v for k, v in config.items() if k not in ("block_type", "type", "block_id")} (или весь config без только type-полей — по соглашению). По соглашению: передавать rest конфига как config блока; block_id при необходимости вынимать отдельно.

### 6.3 Глобальный и локальные реестры

- Для фазы 1 достаточно **одного глобального реестра** (синглтон): BlockRegistry.global_registry() возвращает один и тот же экземпляр. Либо фабрика get_registry() / default_registry().
- Опционально: конструктор BlockRegistry() создаёт пустой локальный реестр; тесты могут создавать свой экземпляр, чтобы не засорять глобальный.

### 6.4 Декоратор register_block

- **register_block(block_type: str, registry: Optional[BlockRegistry] = None)** — декоратор для класса: после определения класса регистрирует его в реестре под block_type. Использование: @register_block("my_block") class MyBlock(AbstractBaseBlock): ...
- Если registry не передан, используется глобальный реестр.

### 6.5 Сигнатуры (псевдокод)

```python
class BlockRegistry:
    @classmethod
    def global_registry(cls) -> "BlockRegistry": ...

    def __init__(self) -> None: ...

    def register(self, block_type: str, factory: Union[Type[AbstractBaseBlock], Callable[..., AbstractBaseBlock]]) -> None: ...

    def build(self, config: Dict[str, Any]) -> AbstractBaseBlock: ...

    def get(self, block_type: str) -> Optional[Union[Type[AbstractBaseBlock], Callable[..., AbstractBaseBlock]]]: ...

    def __contains__(self, block_type: str) -> bool: ...

def register_block(block_type: str, registry: Optional[BlockRegistry] = None) -> Callable: ...
```

### 6.6 Тесты для Registry

Расположение: `tests/foundation/test_registry.py`.

| Тест | Что проверяет |
|------|-------------------------------|
| register затем build(config) | Экземпляр правильного типа; block_type и block_id из config. |
| build с ключом "type" вместо "block_type" | Работает так же. |
| build без block_type и без type в config | KeyError. |
| build с неизвестным block_type | KeyError с сообщением о зарегистрированных типах. |
| get(block_type) после register | Возвращает зарегистрированную фабрику. |
| block_type in registry | True для зарегистрированного, False для нет. |
| Декоратор @register_block("name"): класс затем registry.build({"block_type": "name"}) | Создаётся экземпляр этого класса. |
| Локальный реестр: зарегистрировать в нём, build из него | Работает; глобальный реестр не затронут (если используется отдельный экземпляр в тесте). |

**Критерий прохождения:** все тесты в test_registry.py зелёные.

---

## 7. Вспомогательные блоки для тестов (helpers)

Расположение: `tests/foundation/helpers.py` (или conftest.py с фикстурами). Не входят в пакет yggdrasill; используются только в тестах.

Рекомендуемые классы:

1. **IdentityBlock** — один вход "x", один выход "y"; forward: y = x. Без состояния (state_dict пустой или минимальный). Для проверки блока и реестра.
2. **AddBlock** — входы "a", "b"; выход "out"; forward: out = a + b + offset. offset из config (по умолчанию 0); state_dict() возвращает {"offset": self.offset}; load_state_dict восстанавливает offset. Для проверки конфига, state_dict/load_state_dict и реестра.
3. **Минимальный узёл-задача для тестов** (например IdentityTaskNode): класс, наследующий **и** AbstractBaseBlock, **и** AbstractGraphNode (двойное наследование), с node_id в конструкторе и forward(inputs)=passthrough. Используется в test_node.py для проверки интерфейса Node (get_input_ports, get_output_ports, run); не подменять обёрткой «Node(block)».

Опционально: блок с опциональным портом; блок с get_sub_blocks (вложенный AddBlock) для теста агрегации state_dict.

---

## 8. Порядок реализации (рекомендуемый)

1. **Port** — port.py, тесты test_port.py. Без зависимостей от Block.
2. **AbstractBaseBlock** — block.py; зависит от Port. Тесты test_block.py + helpers (AddBlock, IdentityBlock).
3. **AbstractGraphNode** — node.py; при двойном наследовании использует Block-интерфейс того же объекта. Тесты test_node.py используют минимальный узёл-задачу из helpers (класс Block+Node).
4. **BlockRegistry** — registry.py; зависит от Block. Тесты test_registry.py (регистрируют AddBlock, IdentityBlock).
5. **Экспорты** — в foundation/__init__.py экспортировать Port, PortDirection, PortType, PortAggregation, AbstractBaseBlock, AbstractGraphNode, BlockRegistry, register_block.

После каждого шага запускать `pytest tests/foundation/` и исправлять ошибки.

---

## 9. Проверка и приёмочные критерии

### 9.1 Минимальный сценарий «здравствуй, мир»

- Создать **узёл-задачу** для теста: минимальный класс, наследующий **и** AbstractBaseBlock, **и** AbstractGraphNode (двойное наследование), с node_id и реализацией forward (например passthrough). Либо использовать заглушку из фазы 4 (например IdentityBackbone).
- Вызвать node.run({"x": 42}) у этого объекта.
- Ожидание: {"y": 42}. Убедиться, что run делегирует в forward того же объекта (Block-часть). Отдельной обёртки «узел с блоком» не создаём — в граф кладутся только такие объекты (узлы-задачи).

### 9.2 Сценарий с реестром и состоянием

- Зарегистрировать AddBlock под типом "add".
- Создать блок через registry.build({"block_type": "add", "block_id": "a1", "offset": 3}).
- forward({"a": 1, "b": 2}) → {"out": 6}.
- state = block.state_dict(); новый блок build с другим block_id; load_state_dict(state); forward({"a": 0, "b": 0}) → {"out": 3}.

### 9.3 Чек-лист фазы 1

- [ ] Port: создание, валидация name/direction/aggregation, compatible_with, тесты проходят.
- [ ] AbstractBaseBlock: block_type, block_id, config, forward, state_dict, load_state_dict (без портов и без run); тесты с AddBlock и IdentityBlock проходят.
- [ ] AbstractGraphNode: node_id, declare_ports (абстрактный), get_input_ports, get_output_ports, run (→ self.forward); тесты используют узёл-задачу (класс Block+Node); тесты проходят.
- [ ] BlockRegistry: register, build по config с block_type/type, get, __contains__, декоратор register_block; тесты проходят.
- [ ] foundation/__init__.py экспортирует все публичные классы и функции.
- [ ] `pytest tests/foundation/` — все тесты зелёные.
- [ ] Минимальный сценарий «один блок в узле, run» выполняется без ошибок.

---

## 10. Граничные случаи и валидация

| Ситуация | Ожидаемое поведение |
|----------|----------------------|
| Port с пустым name | ValueError при создании. |
| Выходной порт с aggregation != SINGLE | ValueError при создании. |
| Block: forward с отсутствующим обязательным ключом | По контракту реализации: KeyError или значение по умолчанию для optional. |
| Block: forward возвращает ключ, не объявленный как выходной порт | Допустимо для расширяемости; валидатор графа (фаза 2) может проверять. На фазе 1 не обязательно. |
| load_state_dict с пустым state при strict=True | Если блок не ожидает ключей — успех; иначе исключение. |
| Registry.build с config без block_type и type | KeyError. |
| Registry.build с неизвестным block_type | KeyError с понятным сообщением. |
| AbstractGraphNode с node_id "" или "   " | ValueError. |
| Два порта с одним именем в declare_ports() (у узла) | Невалидно; тест может проверять уникальность или документировать как «поведение не определено». |

---

## 11. Связь с последующими фазами

- **Фаза 2 (движок):** движок будет хранить узлы (node_id → AbstractGraphNode), вызывать node.run(inputs); входы собираются по входящим рёбрам из буферов. Порты узлов используются валидатором и планировщиком.
- **Фаза 3 (гиперграф задачи):** гиперграф добавляет узлы (создаёт блоки через реестр по block_type и config), соединяет порты рёбрами; run(hypergraph, inputs) использует движок и узлы.
- **Фаза 4 (узлы-задачи):** семь ролей — классы с двойным наследованием (Block+Node в одном); регистрируются в том же реестре под типами вида "backbone/...", "conjector/...".

На фазе 1 не реализуются: гиперграф, рёбра, буферы, планировщик, валидатор. Только Port, Block, AbstractGraphNode, Registry.

---

## 12. Итог

Фаза 1 даёт четыре сущности фундамента с явными контрактами и тестами:

- **Port** — описание входа/выхода; совместимость типов.
- **AbstractBaseBlock** — хранилище + выполнение + идентичность + сериализация.
- **AbstractGraphNode** — узел гиперграфа: положение и связи (node_id, declare_ports, get_input_ports, get_output_ports, run); порты объявляет узел; run делегирует в self.forward (Block-часть того же объекта).
- **BlockRegistry** — регистрация типов и создание блоков по конфигу.

После выполнения фазы 1 можно переходить к фазе 2 (гиперграфовый движок) и фазе 3 (гиперграф задачи), опираясь на готовые Port, Block, AbstractGraphNode и Registry.
