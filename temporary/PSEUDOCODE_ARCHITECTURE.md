# Псевдокод архитектуры: два начала, узлы-задачи, гиперграф

**Назначение:** показать, как канон (Block = материальное, Node = идеальное; граф только из узлов-задач; без обёртки) выглядит в коде. Не исполняемый код — схема для реализации по фазам 1–4.

**Язык:** Python-подобный псевдокод.

---

## 1. Два начала (фаза 1)

### 1.1 AbstractBaseBlock — только данные и вычисления

Блок **не знает о портах**: только идентичность, конфиг, состояние и **вычисление** (forward по словарю входов → словарь выходов). Всё, что связано с портами (объявление, типы, связи в графе), относится к **узлу**.

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class AbstractBaseBlock(ABC):
    """
    Материальное начало: идентичность, данные, вычисление.
    Не объявляет порты — это ответственность узла (связи в графе).
    """

    def __init__(self, block_id: str | None = None, *, config: dict | None = None):
        self._block_id = block_id or self._default_block_id()
        self._config = dict(config or {})

    @property
    def block_type(self) -> str:
        raise NotImplementedError

    @property
    def block_id(self) -> str:
        return self._block_id

    @property
    def config(self) -> dict:
        return self._config.copy()

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Вычисление: словарь входов → словарь выходов. Контракт имён ключей задаётся узлом (declare_ports)."""
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        pass
```

### 1.2 AbstractGraphNode — положение, связи и порты

Порты относятся к **идеальному** началу: это точки связей узла в графе. Узел объявляет порты (declare_ports), даёт движку get_input_ports/get_output_ports и run (→ forward того же объекта).

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class AbstractGraphNode(ABC):
    """
    Идеальное начало: node_id, порты (связи в графе), интерфейс для движка.
    Не хранит блок. declare_ports() — здесь (узел задаёт, как вершина подключается к графу).
    При двойном наследовании run(inputs) делегирует в self.forward(inputs) (Block-часть).
    """

    def __init__(self, node_id: str):
        if not (node_id and node_id.strip()):
            raise ValueError("node_id must be non-empty")
        self._node_id = node_id.strip()

    @property
    def node_id(self) -> str:
        return self._node_id

    @abstractmethod
    def declare_ports(self) -> List[Port]:
        """Объявление портов узла (входы/выходы для гиперрёбер). Отвечает узел, не блок."""
        pass

    def get_input_ports(self) -> List[Port]:
        return [p for p in self.declare_ports() if p.direction == PortDirection.IN]

    def get_output_ports(self) -> List[Port]:
        return [p for p in self.declare_ports() if p.direction == PortDirection.OUT]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Для движка: выполнить узел. У узла-задачи делегирует в self.forward (Block-часть)."""
        return self.forward(inputs)
```

---

## 2. Узел-задача: двойное наследование (фаза 4)

### 2.1 Абстрактный бэкбон (Block + Node в одном классе)

```python
class AbstractBackbone(AbstractBaseBlock, AbstractGraphNode):
    """
    Один класс — два начала. Один объект — блок и узел.
    Block: block_type, block_id, config, forward, state_dict (без портов).
    Node: node_id, declare_ports, get_input_ports, get_output_ports, run (→ forward).
    """

    def __init__(
        self,
        node_id: str,
        block_id: str | None = None,
        *,
        config: dict | None = None,
    ):
        AbstractBaseBlock.__init__(self, block_id=block_id, config=config)
        AbstractGraphNode.__init__(self, node_id=node_id)

    @property
    def block_type(self) -> str:
        return "backbone"

    # Порты объявляет узел (связи в графе)
    def declare_ports(self) -> List[Port]:
        return [
            Port("in", PortDirection.IN, PortType.ANY),
            Port("out", PortDirection.OUT, PortType.ANY),
        ]

    # Вычисление — блок
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
```

### 2.2 Заглушка IdentityBackbone (конкретная реализация)

```python
class IdentityBackbone(AbstractBackbone):
    """Минимальная реализация: выход = вход."""

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return dict(inputs)  # passthrough
```

### 2.3 Регистрация в реестре

```python
# При импорте или register_all_stubs():
registry = BlockRegistry.global_registry()
registry.register("backbone/identity", IdentityBackbone)
```

---

## 3. Создание узла-задачи и добавление в граф

### 3.1 Реестр возвращает один объект Block+Node

```python
# Конфиг для узла-задачи (node_id передаётся в конструктор узла-задачи)
build_config = {
    "block_type": "backbone/identity",
    "node_id": "A",
    "block_id": "backbone_A",
    "config": {},
}

# Реестр вызывает фабрику (класс IdentityBackbone); фабрика получает build_config
node = registry.build(build_config)
# node — экземпляр IdentityBackbone, т.е. один объект с:
#   node.node_id, node.get_input_ports(), node.get_output_ports(), node.run()  — от Node
#   node.block_type, node.block_id, node.declare_ports(), node.forward(), node.state_dict()  — от Block

# Отдельной обёртки AbstractGraphNode(block=...) нет — node уже и блок, и узел
assert hasattr(node, "node_id")
assert hasattr(node, "forward")
assert node.run({"in": 42}) == node.forward({"in": 42})  # один и тот же объект
```

### 3.2 Hypergraph.add_node (фаза 3)

```python
def add_node(
    self,
    node_id: str,
    block_type: str,
    config: dict | None = None,
    *,
    block_id: str | None = None,
    registry: BlockRegistry | None = None,
    trainable: bool = True,
    **kwargs,
) -> str:
    registry = registry or BlockRegistry.global_registry()
    build_config = {
        "block_type": block_type,
        "node_id": node_id,
        "block_id": block_id,
        **(config or {}),
    }
    node = registry.build(build_config)  # один объект — узёл-задача (Block+Node)
    self._nodes[node_id] = node
    self._node_trainable[node_id] = trainable
    self._execution_version += 1
    return node_id
```

### 3.3 Hypergraph.from_config (фрагмент по узлам)

```python
for nc in config.get("nodes", []):
    node_id = nc["node_id"]
    block_type = nc["block_type"]
    build_config = {
        "block_type": block_type,
        "node_id": node_id,
        "block_id": nc.get("block_id"),
        **nc.get("config", {}),
    }
    node = registry.build(build_config)
    self.add_node(node_id, node)  # низкоуровневый add_node(node_id, node)
    # или напрямую: self._nodes[node_id] = node
```

---

## 3.4 Механизм создания — цепочка для проверки

Проверка: от конфига/вызова до объекта в графе всё идёт через **один** объект (узёл-задача), без обёртки.

| Шаг | Кто/что | Действие |
|-----|---------|----------|
| 1 | Вход | Конфиг узла: `node_id`, `block_type`, опционально `block_id`, `config`. Либо вызов `add_node(node_id, block_type, config, ...)`, либо запись в `config["nodes"][]` при `from_config`. |
| 2 | Hypergraph / from_config | Собирается **build_config** = `{"block_type": ..., "node_id": ..., "block_id": ..., **config}`. |
| 3 | BlockRegistry | `registry.build(build_config)` — по ключу `block_type` берётся **фабрика** (класс узла-задачи, например `IdentityBackbone`). |
| 4 | Фабрика (класс узла-задачи) | Вызывается конструктор с аргументами из build_config: `IdentityBackbone(node_id=..., block_id=..., config=...)`. Внутри вызываются `AbstractBaseBlock.__init__(self, ...)` и `AbstractGraphNode.__init__(self, node_id)`. |
| 5 | Результат build | Возвращается **один объект** — экземпляр класса с двойным наследованием (Block+Node). У него есть `node_id`, `declare_ports`, `get_input_ports`, `get_output_ports`, `run`, `forward`, `block_type`, `block_id`, `state_dict`. |
| 6 | Hypergraph | Этот объект кладётся в `_nodes[node_id] = node`. Никакого второго шага «обернуть в Node» нет. |
| 7 | Движок | При выполнении `node = hypergraph.get_node(node_id)`; `node.run(inputs)` → `node.forward(inputs)` (тот же объект). |

**Важно:** реестр по `block_type` хранит именно **классы узлов-задач** (IdentityBackbone, IdentityConjector, ...), а не «просто блоки». Поэтому `build()` всегда возвращает объект Block+Node. Отдельного пути «создать блок, потом обернуть в узел» нет.

---

## 4. Движок: выполнение (фаза 2)

### 4.1 Хранилище графа

```python
class Hypergraph:
    def __init__(self):
        self._nodes: Dict[str, Any] = {}  # node_id → узел-задача (объект Block+Node)
        self._edges: List[Edge] = []
        self._exposed_inputs: List[Tuple[str, str, str | None]] = []
        self._exposed_outputs: List[Tuple[str, str, str | None]] = []
```

### 4.2 Исполнитель вызывает node.run

```python
def run_step(executor, node_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    node = hypergraph.get_node(node_id)  # node — узёл-задача (IdentityBackbone и т.д.)
    outputs = node.run(inputs)           # node.run → self.forward (того же объекта)
    return outputs
```

### 4.3 Цикл выполнения (упрощённо)

```python
for node_id in plan:  # план от планировщика (топологический порядок)
    inputs = gather_inputs_from_buffers(node_id)
    outputs = node.run(inputs)  # node = self._nodes[node_id]
    write_outputs_to_buffers(node_id, outputs)
```

---

## 5. Сводка: что в коде чего не бывает

| В коде есть | В коде нет |
|-------------|------------|
| `AbstractBaseBlock` — только block_type, block_id, config, forward, state_dict (без портов, без run) | Портов (declare_ports, get_input_ports, get_output_ports) в Block |
| `AbstractGraphNode(node_id)` — node_id, declare_ports, get_input_ports, get_output_ports, run | `AbstractGraphNode(node_id, block=...)` |
| Порты объявляет **узел** (declare_ports в Node); Block только forward(inputs)→outputs | declare_ports или get_input_ports/get_output_ports в Block |
| Классы вида `AbstractBackbone(AbstractBaseBlock, AbstractGraphNode)` | Отдельного «контейнера», куда кладут блок |
| `node = registry.build(build_config)` → один объект Block+Node | `block = registry.build(...); node = AbstractGraphNode(node_id, block)` |
| `hypergraph._nodes[node_id] = node` (node — узёл-задача) | `hypergraph._nodes[node_id] = (node_id, block)` или обёртки над блоком |
| `node.run(inputs)` → `node.forward(inputs)` (тот же объект) | `node.block.run(inputs)` или `node.block.forward(inputs)` |

---

## 6. Минимальный пример «от конфига до run»

```python
# Регистрация заглушек (фаза 4)
register_all_stubs()  # backbone/identity, converter/identity, ...

# Сборка гиперграфа из конфига (фаза 3)
config = {
    "nodes": [
        {"node_id": "A", "block_type": "backbone/identity", "config": {}},
        {"node_id": "B", "block_type": "backbone/identity", "config": {}},
    ],
    "edges": [
        {"source_node": "A", "source_port": "out", "target_node": "B", "target_port": "in"},
    ],
    "exposed_inputs": [{"node_id": "A", "port_name": "in", "name": "x"}],
    "exposed_outputs": [{"node_id": "B", "port_name": "out", "name": "y"}],
}
g = Hypergraph.from_config(config)

# В g._nodes лежат два объекта типа IdentityBackbone (Block+Node)
# У каждого есть node_id, block_type, forward, run, get_input_ports, get_output_ports

# Выполнение (фаза 2)
from yggdrasill.engine.executor import run
outputs = run(g, {"x": 42})
# outputs["y"] == 42
```

---

**Итог.** В коде два базовых класса (Block и Node) не смешиваются; узел не хранит блок. Узлы-задачи — классы с двойным наследованием; реестр строит их по `block_type`; в графе хранятся только такие объекты; выполнение — вызов `node.run(inputs)` у этого объекта.
