# Фаза 5. Сериализация блока и гиперграфа задачи — полный технический план

Детальный технический план по **пятой фазе**: сохранение и загрузка **блока** и **гиперграфа задачи** по принципу «конфиг + чекпоинт». Цель — обеспечить воспроизводимость: при одинаковых конфиге, чекпоинте и входах результат run детерминирован; артефакты (конфиг и чекпоинт) можно сохранять в файлы и восстанавливать из них без потери структуры и весов. Документ опирается на канон (SERIALIZATION, 01 §2.7, 03 §9) и фазы 1 и 3.

**Связь с планом:** [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — Фаза 5.

**Канон:** [documentation/docs/SERIALIZATION.md](../documentation/docs/SERIALIZATION.md), [documentation/docs/01_FOUNDATION.md](../documentation/docs/01_FOUNDATION.md) §2.7, §9, [documentation/docs/03_TASK_HYPERGRAPH.md](../documentation/docs/03_TASK_HYPERGRAPH.md) §9.

**Язык:** русский.

---

## 1. Цель фазы 5

Реализовать **полный цикл сериализации** для уровня блока и уровня гиперграфа задачи:

- **Блок:** получение сериализуемого конфига (block_type, block_id, параметры), сохранение и загрузка **state_dict**; сохранение блока в виде «конфиг + чекпоинт» в файлы и восстановление из них.
- **Гиперграф задачи:** сохранение **конфига структуры** (to_config уже есть в фазе 3) в файл; сохранение **чекпоинта** (агрегат state_dict узлов с учётом дедупликации по block_id) в файл или директорию; загрузка гиперграфа из конфига и чекпоинта (from_config + load_state_dict или единый load); обеспечение roundtrip: save → load → эквивалентный граф и воспроизводимый run.

**Результат фазы 5:** блок и гиперграф задачи можно сохранить (конфиг + чекпоинт) и загрузить; поведение run воспроизводимо при тех же входах; формат конфига и чекпоинта зафиксирован и версионирован (schema_version).

---

## 2. Зависимости

- **Фаза 0:** структура репозитория, pytest. См. [PHASE_0_STRUCTURE.md](PHASE_0_STRUCTURE.md).
- **Фаза 1:** AbstractBaseBlock с атрибутами block_type, block_id, config, методами state_dict(), load_state_dict(); узлы-задачи (Block+Node) имеют тот же контракт. См. [PHASE_1_FOUNDATION.md](PHASE_1_FOUNDATION.md).
- **Фаза 3:** Hypergraph с to_config(), from_config(), state_dict(), load_state_dict(); конфиг содержит nodes, edges, exposed_inputs, exposed_outputs, graph_id, graph_kind, metadata, schema_version. См. [PHASE_3_TASK_HYPERGRAPH.md](PHASE_3_TASK_HYPERGRAPH.md).

Фаза 5 **добавляет** запись и чтение в файловую систему (или поток/байты): save_config, save_checkpoint, load, load_from_checkpoint; для блока — save_block / load_block (конфиг + чекпоинт). Логика state_dict/load_state_dict и to_config/from_config уже заложена в фазах 1 и 3; фаза 5 формализует форматы файлов, API сохранения/загрузки и дедупликацию по block_id при сохранении/загрузке гиперграфа.

---

## 3. Что входит в фазу 5 и что нет

| Входит в фазу 5 | Не входит (другие фазы) |
|-----------------|--------------------------|
| Конфиг блока: способ получить сериализуемый dict (block_type, block_id, config) из блока/узла-задачи. | Сериализация воркфлоу, стадии, мира, вселенной — фазы 6–10. |
| Сохранение/загрузка **одного блока** в файлы (конфиг + чекпоинт): save_block(block, path_or_dir), load_block(path_or_dir, registry?) → block. | Изменение формата конфига гиперграфа (to_config остаётся как в фазе 3). |
| Формат конфига гиперграфа при записи в файл: JSON или YAML, кодировка, schema_version. | Сжатие и шифрование чекпоинтов — по соглашению реализации, не обязательно в фазе 5. |
| Формат чекпоинта гиперграфа: один файл (dict id → state_dict) или директория с файлами по node_id/block_id; дедупликация по block_id. | Версионирование и миграции старых чекпоинтов (минимально: schema_version в конфиге). |
| API гиперграфа: save(path), save_config(path), save_checkpoint(path); load(path, registry?), load_config(path), load_from_checkpoint(path); опционально load(path) = load_config + load_from_checkpoint. | Чекпоинты в облаке, раздача по сети — уровень развёртывания. |
| Загрузка гиперграфа из каталога: конфиг из config.json (или config.yaml), чекпоинт из checkpoint.json или из директории checkpoints/. | |
| Обработка отсутствующего чекпоинта при load: только структура из конфига (веса не загружены) с явным предупреждением или флагом. | |
| Тесты: roundtrip блока; roundtrip гиперграфа (save → load → to_config совпадает, run даёт тот же результат при тех же входах); общий block_id — одна запись в чекпоинте. | |

---

## 4. Сериализация блока

### 4.1 Назначение по канону

Блок (материальное начало) хранит block_type, block_id, config и состояние (state_dict). Всё, что влияет на поведение forward, должно быть явно в конфиге или в state_dict; тогда конфиг + чекпоинт задают воспроизводимое поведение (01 §2.7, SERIALIZATION §2).

### 4.2 Конфиг блока

Сериализуемый конфиг блока — словарь, достаточный для воссоздания экземпляра того же типа с теми же параметрами (без весов). Рекомендуемый состав:

| Поле | Тип | Описание |
|------|-----|----------|
| **block_type** | str | Тип блока (ключ реестра). Обязателен для build(config). |
| **block_id** | str, optional | Идентификатор экземпляра; при общем блоке в графе — один block_id на несколько узлов. |
| **config** | dict | Параметры, передаваемые в конструктор (размерности, пути, флаги). Должен быть JSON-сериализуемым (без несериализуемых объектов). |

Для узла-задачи (Block+Node) при сериализации **гиперграфа** в конфиг узла входят node_id, block_type, block_id, config, trainable (03 §9.1, Phase 3 to_config). Отдельная сериализация **только блока** (без графа) использует только block_type, block_id, config.

**Получение конфига из блока:** у AbstractBaseBlock в фазе 1 есть атрибуты block_type, block_id, config. Метод `get_config() -> Dict[str, Any]` может быть добавлен в фазу 1 или 5: возвращает `{"block_type": self.block_type, "block_id": self.block_id, "config": dict(self.config)}` (копия, чтобы не мутировать). Если в фазе 1 метода нет — в фазе 5 реализовать в базовом блоке или в хелперах сериализации.

### 4.3 Чекпоинт блока

- **state_dict()** — уже в контракте блока (фаза 1): возвращает словарь состояния (веса, буферы, всё необходимое для load_state_dict). Может быть пустым для блоков без обучаемых параметров.
- **load_state_dict(state, strict=True)** — восстанавливает состояние блока из словаря.

Для сохранения в файл state_dict должен быть сериализуемым. В эталонной реализации (Python/PyTorch) часто используют torch.save(state_dict, path) или pickle; для межплатформенности и без зависимости от фреймворка — JSON только для простых типов, иначе бинарный формат (pickle, safetensors, numpy). В фазе 5 зафиксировать: **формат чекпоинта блока** — по соглашению (например, один файл .pt или .pkl для state_dict, либо .json если state_dict сериализуем в JSON).

### 4.4 API сохранения и загрузки блока

**Сохранение:**

- **save_block(block, path: Union[str, Path], *, config_filename: str = "config.json", checkpoint_filename: str = "checkpoint.pt")**  
  - Если `path` — файл: трактовать как директорию с одним конфигом и одним чекпоинтом не рекомендуется; лучше требовать path как директорию.  
  - Если `path` — директория: записать в неё `config.json` (или config_filename) с конфигом блока (get_config() или {block_type, block_id, config}); записать чекпоинт в `checkpoint.pt` (или checkpoint_filename) — содержимое block.state_dict() в выбранном формате.  
  - Создавать директорию, если не существует.

**Загрузка:**

- **load_block(path: Union[str, Path], registry: Optional[BlockRegistry] = None, *, config_filename: str = "config.json", checkpoint_filename: str = "checkpoint.pt") -> AbstractBaseBlock**  
  - Прочитать конфиг из path/config_filename.  
  - По конфигу вызвать registry.build(config) (реестр по умолчанию — глобальный, если registry не передан). Получить экземпляр блока (или узла-задачи; для «только блок» без графа обычно нужен блок; если реестр возвращает узел-задачу, для фазы 5 допустимо считать его блоком с точки зрения state_dict).  
  - Прочитать чекпоинт из path/checkpoint_filename; вызвать block.load_state_dict(loaded_state).  
  - Вернуть блок.

Передача **registry** обязательна для build(config); если не передана — использовать глобальный реестр или выбросить ошибку, если реестр пуст.

### 4.5 Формат конфига блока в файле

Рекомендуемый формат — JSON (или YAML при наличии зависимостей). Кодировка UTF-8. Пример:

```json
{
  "block_type": "backbone/identity",
  "block_id": "main_backbone",
  "config": {}
}
```

Все значения должны быть JSON-сериализуемыми (числа, строки, списки, словари, булевы, null).

### 4.6 Граничные случаи (блок)

- **Блок без состояния:** state_dict() возвращает {}. При save_block записать пустой чекпоинт или не записывать файл чекпоинта (и при load не требовать его наличия).  
- **Блок с get_sub_blocks:** state_dict по канону может включать вложенные state с префиксами; load_state_dict должен корректно раздавать по подблокам. Сохранение/загрузка одного блока с подблоками — тот же API: один конфиг, один чекпоинт (агрегат).  
- **Несериализуемый config:** при записи конфига в JSON выбросить понятную ошибку (например, TypeError с указанием ключа). Документировать требование JSON-сериализуемости config.

---

## 5. Сериализация гиперграфа задачи

### 5.1 Назначение по канону

Гиперграф задачи сериализуется как **конфиг структуры** (nodes, edges, exposed_inputs, exposed_outputs, graph_id, graph_kind, metadata, schema_version) и **чекпоинт** — словарь идентификаторов → state_dict узлов (с дедупликацией по block_id). Сохранить = записать конфиг + чекпоинт; загрузить = from_config + загрузка чекпоинта в узлы (SERIALIZATION §2, 03 §9).

### 5.2 Конфиг гиперграфа (напоминание из фазы 3)

Структура конфига задана в PHASE_3 §7 и 03 §4.4, §9.1:

- **schema_version** — строка, например "1.0", для обратной совместимости.
- **graph_id** — идентификатор гиперграфа.
- **nodes** — список dict: node_id, block_type, block_id (optional), config, trainable (optional).
- **edges** — список dict: source_node, source_port, target_node, target_port.
- **exposed_inputs** — список dict: node_id, port_name, name (optional).
- **exposed_outputs** — список dict: node_id, port_name, name (optional).
- **graph_kind** — optional.
- **metadata** — optional dict (в т.ч. num_loop_steps).

Метод **to_config()** гиперграфа уже возвращает такой словарь. В фазе 5 этот словарь записывается в файл (JSON или YAML).

### 5.3 Чекпоинт гиперграфа: ключи и дедупликация

- **Ключ по умолчанию:** node_id. state_dict гиперграфа (фаза 3) — агрегат по узлам: {node_id: node.state_dict() for ...}.
- **Дедупликация по block_id:** если несколько узлов разделяют один блок (один block_id, один экземпляр в памяти), в чекпоинте должна храниться **одна** запись по block_id (или каноническому идентификатору общего блока), чтобы не дублировать веса (03 §9.2, SERIALIZATION §4.4). При сохранении: при формировании словаря для чекпоинта для каждого узла смотреть block_id; если block_id совпадает у нескольких узлов, записать state_dict один раз под ключом block_id (или под первым node_id, с таблицей node_id → block_id при загрузке). При загрузке: после from_config узлы уже созданы; при load_state_dict(state) для каждого ключа в state передать state[key] в соответствующий узел; для общего block_id — один и тот же state_dict загрузить в единственный общий экземпляр (все узлы с этим block_id ссылаются на него).

Алгоритм сохранения чекпоинта с дедупликацией:

1. Собрать множество уникальных «единиц загрузки»: либо по node_id (если block_id не используется), либо по block_id когда он задан, иначе по node_id.
2. Для каждой единицы взять state_dict соответствующего узла/блока один раз.
3. Записать в чекпоинт словарь: ключ = block_id если задан и общий, иначе node_id; значение = state_dict.

Алгоритм загрузки:

1. Прочитать чекпоинт (словарь key → state_dict).
2. Для каждого node_id в графе: определить ключ — если у узла есть block_id и в чекпоинте есть запись по этому block_id, использовать block_id; иначе node_id.
3. Вызвать node.load_state_dict(state[key]) для каждого узла (общий block_id приведёт к одному и тому же объекту — загрузка один раз в общий экземпляр).

### 5.4 Форматы файлов

**Конфиг гиперграфа:**

- Один файл: `config.json` (JSON, UTF-8) или `config.yaml` (YAML). Имена полей — как в to_config().

**Чекпоинт гиперграфа:**

- **Вариант A (один файл):** один файл, например `checkpoint.json` или `checkpoint.pt`. Содержимое — словарь {key: state_dict}. Для state_dict с тензорами/бинарными данными JSON не подходит; использовать бинарный формат (torch.save, pickle, safetensors) или раздельно: метаданные в JSON, тензоры в отдельные файлы.
- **Вариант B (директория):** директория `checkpoints/` (или заданное имя); внутри файлы по ключам: `node_1.pt`, `node_2.pt` или `block_id_shared.pt`. Плюс опционально `manifest.json` со списком ключей и соответствием node_id → key.

В фазе 5 зафиксировать один основной вариант (например, один файл checkpoint.pt для всего чекпоинта гиперграфа как словарь) и опционально поддержать загрузку из директории по манифесту.

### 5.5 API сохранения и загрузки гиперграфа

**Сохранение:**

- **hypergraph.save(path: Union[str, Path], *, config_filename: str = "config.json", checkpoint_filename: str = "checkpoint.pt")**  
  - Если path — директория: записать в неё config с помощью to_config() в config_filename; записать чекпоинт (state_dict с дедупликацией по block_id) в checkpoint_filename.  
  - Если path — файл: не поддерживать или трактовать как путь к директории с именем path (не рекомендуется). Лучше явно требовать директорию.

- **hypergraph.save_config(path: Union[str, Path], *, filename: str = "config.json")**  
  - Записать только конфиг (to_config()) в path (файл или директория; если директория — path/config.json).

- **hypergraph.save_checkpoint(path: Union[str, Path], *, filename: str = "checkpoint.pt")**  
  - Записать только чекпоинт (словарь с дедупликацией) в указанный файл (или path/checkpoint.pt если path — директория).

**Загрузка:**

- **Hypergraph.load(path: Union[str, Path], registry: Optional[BlockRegistry] = None, validate: bool = True, *, config_filename: str = "config.json", checkpoint_filename: str = "checkpoint.pt", load_checkpoint: bool = True) -> Hypergraph**  
  - Класс-метод или статический метод. Прочитать конфиг из path/config_filename; вызвать from_config(config, registry=registry, validate=validate); получить гиперграф. Если load_checkpoint=True, прочитать чекпоинт из path/checkpoint_filename и вызвать hypergraph.load_state_dict(loaded_checkpoint). Если файл чекпоинта отсутствует — либо выбросить ошибку, либо загрузить без весов и вернуть гиперграф с незагруженными весами (и опционально предупреждение). Вернуть экземпляр Hypergraph.

- **Hypergraph.load_config(path: Union[str, Path], registry: Optional[BlockRegistry] = None, validate: bool = True, *, config_filename: str = "config.json") -> Hypergraph**  
  - Загрузить только структуру: прочитать конфиг, from_config, вернуть гиперграф без загрузки чекпоинта.

- **hypergraph.load_from_checkpoint(path: Union[str, Path], *, checkpoint_filename: str = "checkpoint.pt")**  
  - Метод экземпляра: прочитать чекпоинт из path (или path/checkpoint_filename) и вызвать self.load_state_dict(loaded). Используется когда граф уже собран (например, из load_config или программно), а веса подгружаются отдельно.

Сигнатуры (псевдокод):

```python
# Сохранение
def save(self, path: Union[str, Path], *, config_filename: str = "config.json", checkpoint_filename: str = "checkpoint.pt") -> None: ...

def save_config(self, path: Union[str, Path], *, filename: str = "config.json") -> None: ...

def save_checkpoint(self, path: Union[str, Path], *, filename: str = "checkpoint.pt") -> None: ...

# Загрузка (класс-методы или модульные функции)
@classmethod
def load(cls, path: Union[str, Path], registry: Optional[BlockRegistry] = None, validate: bool = True, *, config_filename: str = "config.json", checkpoint_filename: str = "checkpoint.pt", load_checkpoint: bool = True) -> "Hypergraph": ...

@classmethod
def load_config(cls, path: Union[str, Path], registry: Optional[BlockRegistry] = None, validate: bool = True, *, config_filename: str = "config.json") -> "Hypergraph": ...

def load_from_checkpoint(self, path: Union[str, Path], *, checkpoint_filename: str = "checkpoint.pt") -> None: ...
```

### 5.6 Размещение кода

- **Модуль сериализации блока:** например `yggdrasill/foundation/serialization.py` или `yggdrasill/serialization/block_io.py` — функции save_block, load_block, при необходимости get_config для блока.
- **Сериализация гиперграфа:** расширить класс Hypergraph в `yggdrasill/engine/structure.py` (или в `yggdrasill/hypergraph/`) методами save, save_config, save_checkpoint, load, load_config, load_from_checkpoint; либо вынести в `yggdrasill/hypergraph/io.py` как функции, принимающие гиперграф и path.
- **Общие утилиты:** запись/чтение JSON с правильной кодировкой, запись/чтение бинарного чекпоинта (torch.save/load или pickle) — в одном месте, чтобы формат был единообразным.

---

## 6. Версионирование и обратная совместимость

- **schema_version** в конфиге гиперграфа (и при необходимости в конфиге блока) задаёт версию формата. При изменении формата в будущем значение увеличивается; загрузчик может проверять версию и применять миграции или отказывать в загрузке неподдерживаемой версии.
- В фазе 5 достаточно одной версии, например "1.0". Документировать: при добавлении полей в конфиг старые конфиги без этих полей считаются совместимыми (значения по умолчанию); при удалении или переименовании полей — новая schema_version и при необходимости миграция.
- Обратная совместимость по канону: по возможности сохранять загрузку старых конфигов и чекпоинтов (SERIALIZATION, философия §1.30).

---

## 7. Воспроизводимость

После загрузки гиперграфа из конфига и чекпоинта вызов **run(hypergraph, inputs)** с теми же inputs и опциями должен давать тот же результат (в пределах контракта блоков и явной стохастичности). Обеспечивается тем, что:

- Структура графа восстанавливается из конфига (from_config).
- Веса восстанавливаются из чекпоинта (load_state_dict).
- В конфиге и чекпоинте нет скрытого состояния; всё, что влияет на run, явно присутствует (01, философия).

В тестах фазы 5 проверять: после save → load конфиг загруженного графа совпадает с to_config() исходного (или эквивалентен); после save → load → run(inputs) выходы совпадают с run(inputs) до сохранения (для детерминированных блоков).

---

## 8. Граничные случаи и ошибки

- **Пустой гиперграф (ноль узлов):** to_config() возвращает конфиг с пустыми nodes, edges; state_dict() — пустой словарь. save сохраняет такой конфиг и пустой чекпоинт; load восстанавливает пустой граф.
- **Гиперграф с одним узлом:** одна запись в nodes и в чекпоинте; без рёбер или с одним узлом и внешними входами/выходами.
- **Отсутствует файл чекпоинта при load(..., load_checkpoint=True):** либо FileNotFoundError с понятным сообщением, либо загрузка без весов с предупреждением (и флаг в объекте или возвращаемом значении). Выбор за реализацией; документировать.
- **Повреждённый или несовместимый конфиг/чекпоинт:** при чтении JSON — явная ошибка парсинга; при load_state_dict — ошибка от блока (strict=True) или логирование и частичная загрузка (strict=False). Не «молча» подменять данные.
- **Разные block_type в конфиге узла и в реестре:** при from_config build(config) вызовет реестр; если block_type не зарегистрирован — ошибка реестра (KeyError или аналог). Сериализация не меняет этого поведения.
- **Путь — не существующая директория при save:** создавать директорию рекурсивно (path.mkdir(parents=True, exist_ok=True)) или требовать существование — зафиксировать в спецификации.

---

## 9. Тесты

### 9.1 Блок

- **Расположение:** `tests/foundation/test_block_serialization.py` или `tests/serialization/test_block_io.py`.
- **Сценарии:**  
  - save_block(block, dir) создаёт config.json и checkpoint.*; load_block(dir, registry) возвращает блок; block.get_config() (или атрибуты) совпадает с записанным конфигом; load_state_dict восстановлен — повторный forward с теми же входами даёт тот же результат.  
  - Блок без состояния (пустой state_dict): после save/load поведение сохраняется.  
  - Ошибки: load_block в несуществующую директорию; конфиг без block_type; реестр без нужного block_type.

### 9.2 Гиперграф задачи

- **Расположение:** `tests/engine/test_hypergraph_serialization.py` или `tests/serialization/test_hypergraph_io.py`.
- **Сценарии:**  
  - **Roundtrip:** g.save(dir); g2 = Hypergraph.load(dir, registry=...); g2.to_config() эквивалентен g.to_config() (сравнение ключей и значений, без идентичности объектов).  
  - **Roundtrip run:** для детерминированного графа (например, заглушки) inputs = {...}; out1 = g.run(inputs); g.save(dir); g2 = Hypergraph.load(dir, registry=...); out2 = g2.run(inputs); assert out1 == out2 (или совпадение по допустимой метрике).  
  - **Дедупликация block_id:** граф с двумя узлами с одинаковым block_id; после save в чекпоинте одна запись по block_id; после load оба узла используют один экземпляр блока и run даёт согласованный результат.  
  - **Только конфиг:** save_config; load_config — граф без весов; затем load_from_checkpoint — веса подгружены; run после этого корректен.  
  - **Отсутствующий чекпоинт:** load(..., load_checkpoint=True) при отсутствии файла — выбранное поведение (ошибка или предупреждение) проверяется тестом.  
  - **Пустой граф:** save/load пустого гиперграфа не падает; загруженный граф имеет пустой to_config и пустой state_dict.

### 9.3 Критерий прохождения фазы 5

Все тесты сериализации блока и гиперграфа зелёные; roundtrip и воспроизводимость run после load подтверждены; дедупликация по block_id работает; API save/load/save_config/save_checkpoint/load_config/load_from_checkpoint документирован и соответствует канону.

---

## 10. Чек-лист реализации

- [ ] Конфиг блока: get_config() или аналог (в блоке или в io-модуле); JSON-сериализация.
- [ ] save_block(block, path), load_block(path, registry); форматы config.json и checkpoint.*.
- [ ] Гиперграф: save(path), save_config(path), save_checkpoint(path).
- [ ] Гиперграф: load(path, registry, ...), load_config(path, ...), load_from_checkpoint(path).
- [ ] Чекпоинт гиперграфа с дедупликацией по block_id при сохранении и загрузке.
- [ ] schema_version в конфиге; документирование формата.
- [ ] Обработка отсутствующего чекпоинта и пустого графа.
- [ ] Тесты блока: roundtrip, пустой state_dict.
- [ ] Тесты гиперграфа: roundtrip to_config, roundtrip run, общий block_id, load_config + load_from_checkpoint.
- [ ] Обновить IMPLEMENTATION_PLAN / README ссылкой на PHASE_5_SERIALIZATION.md.

---

## 11. Связь с каноном и другими фазами

| Документ / фаза | Связь |
|-----------------|--------|
| **SERIALIZATION.md** | Принцип «конфиг + чекпоинт» на каждом уровне; загрузка из чекпоинта; дедупликация по block_id; воспроизводимость. |
| **01_FOUNDATION.md** §2.7, §9 | Блок: state_dict, load_state_dict; сериализация гиперграфа — конфиг структуры + чекпоинт по node_id/block_id. |
| **03_TASK_HYPERGRAPH.md** §9 | Конфиг и чекпоинт гиперграфа; одна запись на block_id при общем блоке; сохранение и загрузка. |
| **Фаза 1** | Контракт блока и узла-задачи (state_dict, load_state_dict, config, block_type, block_id). |
| **Фаза 3** | to_config(), from_config(), state_dict(), load_state_dict() гиперграфа; подготовка к записи в файлы. |
| **Фаза 6** | Сериализация воркфлоу будет опираться на тот же принцип: конфиг воркфлоу + чекпоинт как агрегат чекпоинтов гиперграфов-узлов. |

---

**Итог.** Документ задаёт полную спецификацию **фазы 5**: сериализация блока (конфиг + чекпоинт, save_block/load_block) и гиперграфа задачи (save/save_config/save_checkpoint, load/load_config/load_from_checkpoint, дедупликация по block_id, форматы файлов, версионирование, воспроизводимость). Реализация опирается на контракты фаз 1 и 3 и обеспечивает сохранение и загрузку артефактов без потери структуры и весов.
