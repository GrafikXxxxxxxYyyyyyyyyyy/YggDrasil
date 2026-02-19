# DONE 07: Будущее и улучшения (Future) — каталог возможностей

**Канон проекта:** [CANON.md](../WorldGenerator_2.0/CANON.md). **Канон:** [TODO_07_FUTURE_AND_IMPROVEMENTS.md](../WorldGenerator_2.0/TODO_07_FUTURE_AND_IMPROVEMENTS.md); [HYPERGRAPH_AND_HYPERLINKS.md](../WorldGenerator_2.0/HYPERGRAPH_AND_HYPERLINKS.md) (гиперграф как движок, гиперссылки); [END_WORLDS.md](../WorldGenerator_2.0/END_WORLDS.md) (фиксированные миры).

**Статус:** TODO_07 — не обязательный план, а **каталог возможностей**. Реализация по пунктам — по приоритету и ресурсам.

---

## 1. Что уже зафиксировано в каноне

- **Модели через API** (раздел 1.6 в TODO_07): [LLM_API_SUPPORT.md](../WorldGenerator_2.0/LLM_API_SUPPORT.md) — поддержка API-провайдеров для **любого типа блока** (LLM, VLM, эмбеддинги, генерация по API и т.д.), для которого у провайдера есть API; без локальных весов.
- **Повторное использование одной модели** (раздел 1.7 в TODO_07): [MODEL_REUSE.md](../WorldGenerator_2.0/MODEL_REUSE.md) — один checkpoint_ref / model_id → один экземпляр на всех уровнях иерархии.
- **Обучение при повторном использовании и при API:** [TRAINING_REUSE_AND_API_SCENARIOS.md](../WorldGenerator_2.0/TRAINING_REUSE_AND_API_SCENARIOS.md) — коллизии (оптимизатор, градиенты), LoRA при общем блоке, обучение мира с API-узлами (промпт-адаптеры), сценарии и обходы.
- **Поддержка агентных систем:** [AGENT_SYSTEMS_SUPPORT.md](../WorldGenerator_2.0/AGENT_SYSTEMS_SUPPORT.md) — полная и глубоко встроенная: блок-агент, роль Agent, инструменты, agent_loop в графе, граф-агент как узел пайплайна/этапа, мир с агентными этапами.

При реализации любых расширений из TODO_07 эти правила соблюдаются.

---

## 2. Что не реализовано (по TODO_07)

Расширения из каталога: VLM, видео, аудио, мультимодальность, стриминг, бэкенды чекпоинтов (torch, safetensors), асинхронность, трейсинг, multi-endpoint развёртывание, LoRA и адаптеры, и т.д. — полный список в [TODO_07_FUTURE_AND_IMPROVEMENTS.md](../WorldGenerator_2.0/TODO_07_FUTURE_AND_IMPROVEMENTS.md).

---

## 3. Что нужно реализовать

По мере приоритета — отдельные пункты из TODO_07. При реализации: одинаковые модели (один ref/model_id) — один экземпляр в памяти; при обучении и LoRA с повторным использованием и API — следовать [TRAINING_REUSE_AND_API_SCENARIOS.md](../WorldGenerator_2.0/TRAINING_REUSE_AND_API_SCENARIOS.md).

Сводка по этапам: [СТАТУС_ПО_ЭТАПАМ.md](СТАТУС_ПО_ЭТАПАМ.md).
