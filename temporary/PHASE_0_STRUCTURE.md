# Фаза 0. Структура репозитория — полная спецификация

Детальный документ по **нулевой фазе**: всё, что нужно сделать для подготовки репозитория к разработке кодовой базы фреймворка Иггдрасиль. Тесты — в отдельной папке в корне; весь код фреймворка — в пакете в корне; конфигурация сборки, зависимости и окружение задаются явно.

**Связь с планом:** [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — Фаза 0.

**Канон:** [documentation/Scheme.md](../documentation/Scheme.md), философия проекта (documentation опирается на неё).

**Язык:** русский.

---

## 1. Цель фазы 0

- Зафиксировать **расположение кода** и **тестов** в репозитории.
- Ввести **единую точку сборки и установки** пакета (pyproject.toml, при необходимости setup.py / setup.cfg).
- Задать **зависимости** (runtime и для разработки/тестов) и способ их установки (requirements.txt, optional-dependencies в pyproject.toml).
- Обеспечить **запуск тестов** из корня проекта без дополнительной магии.
- Подготовить **структуру каталогов** под последующие фазы (фундамент, движок, гиперграф, воркфлоу и т.д.).

**Результат фазы 0:** после выполнения всех пунктов можно выполнить `pip install -e .` в корне, импортировать `yggdrasill` и запустить `pytest tests/`; структура готова к фазе 1.

---

## 2. Расположение директорий в корне проекта

Корень проекта — каталог, в котором лежат `documentation/`, `philosophy/`, `temporary/` и куда добавляются каталоги кода и тестов.

| Путь от корня | Назначение |
|---------------|------------|
| **yggdrasill/** | Пакет с **всей кодовой базой фреймворка**. Импорт: `import yggdrasill` или `from yggdrasill.foundation import ...`. Внутри — подпакеты по уровням (foundation, task_nodes, hypergraph, workflow, stage, world, universe и т.д.) по мере реализации. |
| **tests/** | **Все тесты проекта.** Отдельная папка в корне (не внутри yggdrasill). Структура зеркалирует или группирует тесты по уровням: foundation, task_nodes, hypergraph, workflow и т.д. Запуск: `pytest tests/` из корня. |
| **documentation/** | Канон и документация (уже есть). Не трогать в фазе 0. |
| **philosophy/** | Философия проекта (уже есть). |
| **temporary/** | Планирование разработки, TODO, лог, план фаз (уже есть). |

Итог: **код — в `yggdrasill/`**, **тесты — в `tests/`**, оба в **корне** репозитория.

---

## 3. Структура каталога `yggdrasill/` (пакет фреймворка)

На момент фазы 0 пакет должен быть **импортируемым** и содержать заглушки под будущие модули, чтобы не ломать импорты и тесты.

### 3.1 Дерево каталогов (целевое для фазы 0)

```
yggdrasill/
├── __init__.py          # Версия и публичный API пакета (можно минимальный)
├── foundation/          # Фундамент: Block, Node, Port, Registry (фаза 1)
│   └── __init__.py
├── task_nodes/          # Абстрактные узлы-задачи, семь ролей (фаза 4)
│   └── __init__.py
├── engine/              # Гиперграфовый движок (фаза 2)
│   └── __init__.py
├── hypergraph/          # Гиперграф задачи (фаза 3)
│   └── __init__.py
├── workflow/            # Воркфлоу (фаза 6)
│   └── __init__.py
├── stage/               # Стадия (фаза 7)
│   └── __init__.py
├── world/               # Мир (фаза 8)
│   └── __init__.py
├── universe/            # Вселенная (фаза 9)
│   └── __init__.py
└── py.typed             # Опционально: маркер для PEP 561 (типизированный пакет)
```

Все `__init__.py` на фазе 0 могут быть пустыми или с кратким docstring; подпакеты пока не экспортируют классы (они появятся в следующих фазах).

### 3.2 Содержимое `yggdrasill/__init__.py`

Минимум: версия пакета и при необходимости список публичных имён `__all__`. Пример:

```python
"""Yggdrasill — фреймворк гиперграфов от блока до вселенной (канон documentation/)."""

__version__ = "0.1.0"
```

Остальные `yggdrasill/*/__init__.py` — пустые или с однострочным docstring.

---

## 4. Структура каталога `tests/`

Тесты лежат **в корне проекта** в папке **tests/** (отдельно от кода). Структура закладывается под уровни канона и движок.

### 4.1 Дерево каталогов (целевое для фазы 0)

```
tests/
├── __init__.py          # Пустой или с пометкой, что это тесты
├── conftest.py          # Общие фикстуры pytest (пустой или минимальный)
├── foundation/          # Тесты фундамента: Block, Node, Port, Registry
│   └── __init__.py
├── task_nodes/          # Тесты узлов-задач
│   └── __init__.py
├── engine/              # Тесты движка (Validator, Planner, Executor, Buffers)
│   └── __init__.py
├── hypergraph/          # Тесты гиперграфа задачи
│   └── __init__.py
├── workflow/            # Тесты воркфлоу
│   └── __init__.py
├── stage/               # Тесты стадии
│   └── __init__.py
├── world/               # Тесты мира
│   └── __init__.py
└── universe/            # Тесты вселенной
    └── __init__.py
```

На фазе 0 в каждой подпапке — только `__init__.py` (пустой). Реальные тестовые модули (`test_*.py`) добавляются в соответствующих фазах.

### 4.2 Запуск тестов

- Из **корня проекта**: `pytest tests/`
- Только один подпакет: `pytest tests/foundation/`
- С выводом print: `pytest tests/ -v -s`
- Покрытие (если настроено): `pytest tests/ --cov=yggdrasill`

Для этого в корне должны быть установлены pytest и при необходимости pytest-cov; зависимости для тестов задаются в pyproject.toml (optional-dependencies dev) и/или в requirements-dev.txt.

---

## 5. Файлы конфигурации в корне проекта

В корне проекта должны появиться (или быть актуализированы) следующие файлы.

### 5.1 `pyproject.toml`

Единая точка метаданных проекта, зависимостей и настроек инструментов (pytest, ruff, mypy и т.д.). Сборка пакета выполняется через этот файл (PEP 517/518).

**Обязательные секции для фазы 0:**

- **[build-system]** — backend сборки (setuptools).
- **[project]** — name, version, description, readme, requires-python, license, authors; dependencies (минимальный набор для старта).
- **[project.optional-dependencies]** — группа **dev**: pytest, pytest-cov (опционально), ruff, mypy (опционально).
- **[tool.setuptools.packages.find]** — указать, что пакет находится в корне: `where = ["."]`, `include = ["yggdrasill", "yggdrasill*"]`.

**Рекомендуемые поля [project]:**

- `name`: `"yggdrasill"` (имя пакета на PyPI; в репозитории может быть YggDrasill).
- `version`: например `"0.1.0"` (старт разработки).
- `description`: краткое описание фреймворка по канону.
- `readme`: `"README.md"` (если в корне есть README).
- `requires-python`: `">=3.10"` (или другой минимум по решению).
- `license`: текст или файл (например, `{text = "Apache-2.0"}`).
- `dependencies`: на фазе 0 можно оставить пустой список или минимальный (например, только для типов/тестов). Тяжёлые зависимости (torch, diffusers и т.д.) добавлять по мере фаз.

**Опциональные зависимости (dev):**

- `pytest>=7.0.0`
- `pytest-cov` — по желанию
- `ruff>=0.1.0` — линтер/форматтер
- `mypy>=1.0.0` — по желанию

**Скрипты/точки входа:** при появлении CLI в следующих фазах — `[project.scripts]` с записью вида `yggdrasill = "yggdrasill.cli:main"`.

Пример минимального `pyproject.toml` для фазы 0 см. в разделе 8.

### 5.2 `setup.py` (опционально)

Современный подход — собирать только через `pyproject.toml`. Если нужна совместимость со старыми `pip`/`setup.py install`, можно оставить тонкую обёртку:

```python
from setuptools import setup
setup()
```

Вся конфигурация при этом в `pyproject.toml`. Либо не создавать `setup.py` вовсе и полагаться на `pip install -e .` с pyproject.toml.

### 5.3 `setup.cfg` (опционально)

Если часть настроек хочется хранить в setup.cfg (например, для setuptools), можно добавить секции `[options]`, `[options.packages.find]` и т.д. Для фазы 0 достаточно одного `pyproject.toml`; `setup.cfg` можно не вводить.

### 5.4 `requirements.txt`

Используется для установки зависимостей без установки самого пакета в режиме разработки (например, в CI или в виртуальном окружении только с зависимостями).

**Два варианта:**

1. **Минимальный** — перечислить только то, что нужно для запуска тестов и линтеров (pytest, ruff и т.д.). Пример:
   ```
   pytest>=7.0.0
   pytest-cov>=4.0.0
   ruff>=0.1.0
   ```

2. **Ссылка на pyproject** — в README или в комментарии указать: «Установка зависимостей для разработки: `pip install -e ".[dev]"`» и не вести отдельный requirements.txt, либо генерировать его из pyproject (например, `pip freeze` после `pip install -e ".[dev]"` в отдельном venv).

**Рекомендация для фазы 0:** создать `requirements-dev.txt` (или `requirements.txt`) с зависимостями для разработки и тестов, чтобы одной командой `pip install -r requirements-dev.txt` поднять окружение; сам пакет тогда ставить `pip install -e .`.

### 5.5 `README.md` в корне (опционально для фазы 0)

Если в корне ещё нет README, можно добавить краткий: название проекта, что это (фреймворк по канону documentation/), как установить (`pip install -e .`), как запустить тесты (`pytest tests/`), ссылка на documentation/ и temporary/IMPLEMENTATION_PLAN.md. Не обязательно для «прохождения» фазы 0, но полезно для навигации.

### 5.6 `.gitignore`

Уже есть. Проверить, что в нём игнорируются:

- `build/`, `dist/`, `*.egg-info/`, `eggs/`, `*.egg`
- `.venv/`, `venv/`, `env/`
- `__pycache__/`, `.pytest_cache/`, `.coverage`, `htmlcov/`
- `.mypy_cache/`, `.ruff_cache/`

При необходимости добавить недостающее. Отдельно игнорировать папку `temporary/` в .gitignore **не** нужно — она часть репозитория до завершения разработки.

---

## 6. Чек-лист выполнения фазы 0

Выполнять по порядку.

### Шаг 1. Создать пакет `yggdrasill/` в корне

- [ ] Создать каталог `yggdrasill/` в корне проекта.
- [ ] Создать `yggdrasill/__init__.py` с `__version__ = "0.1.0"` (и при необходимости кратким docstring).
- [ ] Создать подкаталоги: `foundation/`, `task_nodes/`, `engine/`, `hypergraph/`, `workflow/`, `stage/`, `world/`, `universe/`.
- [ ] В каждом подкаталоге создать `__init__.py` (пустой или с однострочным docstring).
- [ ] Опционально: добавить `yggdrasill/py.typed` (пустой файл) для типизированного пакета.

### Шаг 2. Создать каталог тестов `tests/` в корне

- [ ] Создать каталог `tests/` в корне проекта.
- [ ] Создать `tests/__init__.py` (пустой).
- [ ] Создать `tests/conftest.py` (пустой или с одной фикстурой-заглушкой).
- [ ] Создать подкаталоги: `tests/foundation/`, `tests/task_nodes/`, `tests/engine/`, `tests/hypergraph/`, `tests/workflow/`, `tests/stage/`, `tests/world/`, `tests/universe/`.
- [ ] В каждом подкаталоге создать `__init__.py` (пустой).

### Шаг 3. Добавить `pyproject.toml` в корень

- [ ] Создать файл `pyproject.toml` в корне (содержимое см. раздел 8 или адаптировать под себя).
- [ ] Указать `name = "yggdrasill"`, `version`, `requires-python`, `dependencies` (минимальные или пустые), `[project.optional-dependencies]` с группой `dev` (pytest, ruff и т.д.).
- [ ] Настроить `[tool.setuptools.packages.find]`: пакет `yggdrasill` в корне.
- [ ] При необходимости добавить `[tool.pytest.ini_options]`, `[tool.ruff]` и т.д.

### Шаг 4. Добавить `setup.py` (по желанию)

- [ ] Либо создать минимальный `setup.py` с одним вызовом `setup()` (всё остальное в pyproject.toml).
- [ ] Либо не создавать и использовать только `pyproject.toml`.

### Шаг 5. Добавить файлы зависимостей

- [ ] Создать `requirements-dev.txt` (или `requirements.txt`) с зависимостями для разработки и тестов: pytest и т.д.
- [ ] При необходимости создать `requirements.txt` с минимальными runtime-зависимостями (на фазе 0 может быть пустым или без тяжёлых библиотек).

### Шаг 6. Проверить `.gitignore`

- [ ] Убедиться, что в .gitignore перечислены артефакты сборки, кэши, виртуальные окружения, отчёты покрытия и т.д.

### Шаг 7. Установка и проверка

- [ ] Создать виртуальное окружение в корне: `python -m venv .venv`, активировать.
- [ ] Установить пакет в режиме разработки: `pip install -e .`
- [ ] Установить зависимости для тестов: `pip install -e ".[dev]"` или `pip install -r requirements-dev.txt`
- [ ] Проверить импорт: `python -c "import yggdrasill; print(yggdrasill.__version__)"`
- [ ] Запустить тесты: `pytest tests/` (должно завершиться без ошибок; тестов может быть 0 — тогда 0 passed).

### Шаг 8. Фиксация в temporary

- [ ] В [TODO.md](TODO.md) отметить задачу «Фаза 0» как выполненную или перенести в [DEV_LOG.md](DEV_LOG.md) запись о выполнении фазы 0.

---

## 7. Конфигурация pytest (в pyproject.toml)

Чтобы pytest искал тесты в `tests/` и использовал корень как корень проекта:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-v"
```

При необходимости добавить `filterwarnings` или `markers`.

---

## 8. Пример минимального `pyproject.toml` для фазы 0

Ниже — пример содержимого `pyproject.toml`, которого достаточно для фазы 0. Имена и версии можно заменить на актуальные.

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "yggdrasill"
version = "0.1.0"
description = "Yggdrasill — фреймворк гиперграфов от блока до вселенной (канон documentation/)"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "Apache-2.0"}
authors = [
    {name = "Yggdrasill Contributors"}
]
keywords = [
    "hypergraph",
    "framework",
    "diffusion",
    "workflow",
    "world-modeling",
]
classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.1.0",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["yggdrasill", "yggdrasill*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-v"

[tool.ruff]
line-length = 120
target-version = "py310"
```

После добавления тяжёлых зависимостей (torch, diffusers и т.д.) их перечисляют в `dependencies` или в отдельных optional-dependencies (например, `full`, `train`).

---

## 9. Пример `requirements-dev.txt`

```
# Зависимости для разработки и тестов (фаза 0).
# Установка пакета: pip install -e .
# Установка dev-зависимостей: pip install -r requirements-dev.txt

pytest>=7.0.0
pytest-cov>=4.0.0
ruff>=0.1.0
```

При использовании только pyproject.toml можно обойтись командой `pip install -e ".[dev]"` и не вести requirements-dev.txt; тогда этот файл опционален.

---

## 10. Итог фазы 0

После выполнения всех шагов:

- В **корне** есть пакет **yggdrasill/** со всеми подпакетами (foundation, task_nodes, engine, hypergraph, workflow, stage, world, universe) и пустыми `__init__.py`.
- В **корне** есть каталог **tests/** с подкаталогами под уровни и пустыми `__init__.py` и `conftest.py`.
- В **корне** есть **pyproject.toml** (и при необходимости setup.py, requirements-dev.txt, requirements.txt), позволяющие установить пакет и запустить тесты.
- Команды `pip install -e .`, `pip install -e ".[dev]"` (или `pip install -r requirements-dev.txt`) и `pytest tests/` выполняются без ошибок.
- Документация и временные файлы планирования остаются в `documentation/` и `temporary/`.

На этом фаза 0 считается завершённой; дальнейшая разработка ведётся по [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) начиная с фазы 1.
