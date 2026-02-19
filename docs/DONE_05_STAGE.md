# DONE 05: Этап (Stage) — не реализован

**Канон:** [WorldGenerator_2.0/TODO_05_STAGE.md](../WorldGenerator_2.0/TODO_05_STAGE.md), [Stage_Level.md](../WorldGenerator_2.0/Stage_Level.md), [Scheme.md](../WorldGenerator_2.0/Scheme.md).

**Статус: этап (Stage) не реализован.** Ниже — что требуется по канону и что нужно реализовать.

---

## 1. Что должно быть (по TODO_05)

- **Stage** — граф, узлами которого являются пайплайны (Pipeline); рёбра этапа соединяют внешние выходы одного пайплайна с внешними входами другого; по рёбрам передаётся **state** (в каноне — контейнер из пяти блоков).
- **Контракт этапа по state:** input_state_spec, output_state_spec, маппинг state ↔ порты пайплайнов; **условия выполнения** по заполненности блоков state (Scheme: Философ при 1,2,3; Автор при 1–4; World update при всех 5; Development безусловно и т.д.).
- **API:** AddPipeline(pipeline_or_config, stage_node_id), add_edge, set_state_contract, set_execution_condition; validate; run(stage, state) с учётом execution_condition.
- **Сериализация:** to_config/from_config, state_dict/load_state_dict, save/load, checkpoints; trainable пайплайны.
- **Учёт канона:** [LLM_API_SUPPORT.md](../WorldGenerator_2.0/LLM_API_SUPPORT.md) — этапы могут использовать пайплайны с моделями через API (любой блок); [MODEL_REUSE.md](../WorldGenerator_2.0/MODEL_REUSE.md) — один checkpoint_ref в разных пайплайнах этапа → один экземпляр модели в памяти; [TRAINING_REUSE_AND_API_SCENARIOS.md](../WorldGenerator_2.0/TRAINING_REUSE_AND_API_SCENARIOS.md) — обучение этапа при повторном использовании и при API-узлах.

---

## 2. Что не сделано

Всё перечисленное выше: структура Stage, рёбра этапа, контракт по state, роль этапа, execution_condition, API построения, валидация, исполнитель run(stage, state), сериализация и загрузка из чекпоинта, from_config/from_template, пул повторного использования по checkpoint_ref.

---

## 3. Что нужно реализовать

Реализовать уровень Stage по [TODO_05_STAGE.md](../WorldGenerator_2.0/TODO_05_STAGE.md) целиком: контейнер пайплайнов, рёбра по внешним портам, контракт по state и условия выполнения (Scheme), run(stage, state), конфиг и чекпоинты; при реализации соблюдать правило «один checkpoint_ref → один экземпляр» на уровне этапа. При обучении этапа с общими моделями и API-узлами — [TRAINING_REUSE_AND_API_SCENARIOS.md](../WorldGenerator_2.0/TRAINING_REUSE_AND_API_SCENARIOS.md).

Сводка по этапам: [СТАТУС_ПО_ЭТАПАМ.md](СТАТУС_ПО_ЭТАПАМ.md).
