# Среда разработки YggDrasil

## Быстрый старт

Репозиторий уже клонирован, виртуальное окружение создано и зависимости установлены.

### Активация окружения

```bash
cd /workspace/YggDrasil
source .venv/bin/activate
```

После активации доступны:
- `python` — интерпретатор с установленным пакетом `yggdrasil`
- `yggdrasil` — CLI (если entry point зарегистрирован)
- `pytest` — запуск тестов
- `ruff` — линтер
- `mypy` — проверка типов

### Проверка установки

```bash
python -c "import yggdrasil; print(yggdrasil.__version__)"
# yggdrasil 0.1.0
```

### Запуск тестов

```bash
pytest tests/ -v
# Без падения на первой ошибке:
pytest tests/ -v --no-fail-fast
```

### Линтинг и типы

```bash
ruff check yggdrasil/
mypy yggdrasil/
```

### Установка заново (editable + full + dev)

```bash
pip install -e ".[full,dev]"
```

## Структура

- `yggdrasil/` — исходный код фреймворка
- `examples/` — примеры: `examples/images/sd15`, `examples/images/sdxl`, `examples/audio`
- `tests/` — тесты
- `configs/` — конфиги и пресеты

## Зависимости

- **Python** ≥ 3.10 (проверено на 3.12)
- **PyTorch** ≥ 2.3, **diffusers**, **transformers**, **accelerate**
- **FastAPI**, **Gradio** — сервер и UI
- **pytest**, **ruff**, **mypy** — для разработки

Подробности в `pyproject.toml`.
