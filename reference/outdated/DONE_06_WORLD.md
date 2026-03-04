# DONE 06: Мир (World) — не реализован

**Канон проекта:** [CANON.md](../WorldGenerator_2.0/CANON.md). **Канон:** [TODO_06_WORLD.md](../WorldGenerator_2.0/TODO_06_WORLD.md), [World_Level.md](../WorldGenerator_2.0/World_Level.md), [Scheme.md](../WorldGenerator_2.0/Scheme.md) (§1.1 роли этапов), [World_Serialization.md](../WorldGenerator_2.0/World_Serialization.md).

**Статус: уровень мира (World) не реализован.** Ниже — что требуется по канону и что нужно реализовать.

---

## 1. Что должно быть (по TODO_06)

- **World** — граф, узлами которого являются этапы (Stage); **цикл** — упорядоченный список stage_id (в каноне: Философ → Автор → Среда → Архитектор → Творец, затем снова первый); state передаётся между этапами по циклу.
- **State и storage:** state_schema (в каноне — пять блоков); хранилище для сохранения state при World update (этап Среда); initial_world (начальное содержание мира).
- **Условия выполнения:** по [Scheme.md](../WorldGenerator_2.0/Scheme.md) §4, §4.8 — для каждого этапа условие по заполненности блоков state; при пустом state первая итерация начинается с Development of the world.
- **API:** AddStage, SetCycle/set_canonical_cycle, set_state_schema, set_storage, set_initial_world; validate; run(world, state, action, num_steps); сохранение state при World update.
- **Сериализация:** to_config/from_config, save/load, checkpoints по этапам; from_template; trainable этапы, LoRA-world.
- **Учёт канона:** [LLM_API_SUPPORT.md](../WorldGenerator_2.0/LLM_API_SUPPORT.md) — мир может использовать модели в этапах через API (любой блок); [MODEL_REUSE.md](../WorldGenerator_2.0/MODEL_REUSE.md) — один checkpoint_ref в разных этапах/пайплайнах/графах → один экземпляр модели в памяти; [TRAINING_REUSE_AND_API_SCENARIOS.md](../WorldGenerator_2.0/TRAINING_REUSE_AND_API_SCENARIOS.md) — обучение мира и LoRA при API-узлах (промпт-адаптеры и т.д.); [AGENT_SYSTEMS_SUPPORT.md](../WorldGenerator_2.0/AGENT_SYSTEMS_SUPPORT.md) — мир может состоять из агентных этапов (action и контекст мира передаются в этапы, напр. Development of the world как агент).

---

## 2. Что не сделано

Всё перечисленное выше: структура World, цикл, state_schema, storage, initial_world, API построения, валидация, исполнитель мира с учётом Scheme §4.8, сохранение state в storage, конфиг и чекпоинты, шаблоны, обучение и LoRA-world; пул по checkpoint_ref при загрузке мира.

---

## 3. Что нужно реализовать

Реализовать уровень World по [TODO_06_WORLD.md](../WorldGenerator_2.0/TODO_06_WORLD.md) целиком: контейнер этапов, цикл, state и storage, условия по Scheme (§4, §4.8), исполнитель run(world, state, action), сериализация и шаблоны; при загрузке соблюдать правило «один checkpoint_ref → один экземпляр». При обучении мира с API-этапами и LoRA — [TRAINING_REUSE_AND_API_SCENARIOS.md](../WorldGenerator_2.0/TRAINING_REUSE_AND_API_SCENARIOS.md).

Сводка по этапам: [СТАТУС_ПО_ЭТАПАМ.md](СТАТУС_ПО_ЭТАПАМ.md).
