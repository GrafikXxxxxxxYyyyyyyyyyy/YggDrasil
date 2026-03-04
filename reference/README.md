# YggDrasill

**YggDrasill** — универсальный Lego-фреймворк для диффузионных моделей и генерации миров (название — от скандинавского мирового древа Иггдрасиль): одна иерархия от блока до мира, один контракт (входы/выходы, run) и сериализация на каждом уровне. Изображения, видео, аудио, LLM — любые модальности и любые модели; мир (цикл этапов и state) — объект первого класса.

**Принцип сериализуемости:** проект должен быть **полностью сериализуем на каждом уровне представления** — от фундамента (блок, узел) до высшей абстракции (мир). Сериализуются **структура и веса** на каждом уровне; для **каждого уровня** должна быть **возможность загрузки из чекпоинта** (блок, граф, пайплайн, этап, мир). Подробно: [WorldGenerator_2.0/SERIALIZATION_AT_ALL_LEVELS.md](WorldGenerator_2.0/SERIALIZATION_AT_ALL_LEVELS.md).

**Развёртывание на нескольких эндпоинтах:** система должна позволять **запуск на нескольких эндпоинтах**; идеально — **каждый граф может быть развёрнут на своём эндпоинте** при полном сохранении корректной работы системы. Реализация инференса должна давать возможность разворачивать каждый граф проекта на отдельном эндпоинте. Подробно: [docs/MULTI_ENDPOINT_DEPLOYMENT.md](docs/MULTI_ENDPOINT_DEPLOYMENT.md).

**Масштабируемость и расширяемость:** проект должен быть **абсолютно масштабируем под любые кастомные решения и реализации**. Поддержка работы с **LLM, VLM, любой диффузионной моделью** и **любой нейросетевой архитектурой**; новые модели и архитектуры подключаются как блоки с контрактом портов без изменения ядра. Подробно: [docs/SCALABILITY_AND_EXTENSIBILITY.md](docs/SCALABILITY_AND_EXTENSIBILITY.md).

**Обучаемость на всех уровнях:** на **каждом уровне представления** проект должен быть **полностью обучаем**. Любой блок, любой узел, любая обучаемая величина — LoRA, ControlNet, Backbone, кодек, кондиционер и т.д. — должна поддерживать обучение на своём уровне (блок, граф, пайплайн, этап, мир). Подробно: [WorldGenerator_2.0/TRAINABILITY_AT_ALL_LEVELS.md](WorldGenerator_2.0/TRAINABILITY_AT_ALL_LEVELS.md).

**LoRA-мир и лёгкие кастомные миры:** механизм, при котором **все веса собранного мира замораживаются**, а ко **всем необходимым частям** (backbone, LLM, conditioner и т.д.) подключаются **LoRA-адаптеры**; обучаются **только адаптеры**. В результате — **лёгкие кастомные миры**: один базовый мир + небольшие обучаемые LoRA, без полного fine-tune. Подробно: [WorldGenerator_2.0/LORA_WORLD_LIGHTWEIGHT_CUSTOM.md](WorldGenerator_2.0/LORA_WORLD_LIGHTWEIGHT_CUSTOM.md).

**Поддержка агентных систем:** фреймворк обеспечивает **полную и глубоко встроенную поддержку агентных систем** — агент как блок/узел (состояние между вызовами, вызов инструментов), роль Agent и узлы-инструменты, режим agent_loop в графе, этапы и мир с агентными реализациями (в т.ч. Development of the world как агент). Подробно: [WorldGenerator_2.0/AGENT_SYSTEMS_SUPPORT.md](WorldGenerator_2.0/AGENT_SYSTEMS_SUPPORT.md).

## Установка

**Из PyPI (после публикации):**
```bash
pip install yggdrasil
```

**Из исходников (клонированный репозиторий):**
```bash
cd YggDrasill
pip install .
```

**Режим разработки (editable):**
```bash
pip install -e .
```

**Из Git:**
```bash
pip install git+https://github.com/your-org/YggDrasill.git
```

**Публикация на PyPI** (чтобы работало `pip install yggdrasil` для всех):
```bash
pip install build twine
python -m build
twine upload dist/*
# или для Test PyPI: twine upload --repository testpypi dist/*
```

После установки доступна команда `yggdrasil`:
```bash
yggdrasil ui          # Gradio-интерфейс
yggdrasil ui --share  # с публичной ссылкой
yggdrasil api         # REST API
python -m yggdrasil ui  # то же через модуль
```

Дополнительные зависимости (обучение, LoRA, графы):
```bash
pip install yggdrasil[full]   # safetensors, peft, torchvision, ...
pip install yggdrasil[train]  # для обучения
pip install yggdrasil[dev]    # pytest, ruff, mypy
```

## Документация

Документация проекта ведётся **только на русском языке**.

- **Канон (архитектура, мир, схема):** [WorldGenerator_2.0/](WorldGenerator_2.0/) — Scheme, Philosophy, System, уровни Graph/Pipeline/Stage/World, сериализация, LLM API, повторное использование моделей, TODO.
- **Реализация (что сделано):** [docs/](docs/) — DONE_01…DONE_04 (Foundation, Task nodes, Graph engine, Pipeline), планы и EXPAND.
- **Визия и дорожная карта документации:** [docs/VISION_AND_DOCUMENTATION_ROADMAP.md](docs/VISION_AND_DOCUMENTATION_ROADMAP.md) — как сделать фреймворк золотым стандартом для диффузии и генерации миров и самым удобным в использовании (что добавить в документацию).