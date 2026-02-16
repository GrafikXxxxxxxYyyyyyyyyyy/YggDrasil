# Примеры пайплайнов

## Комбинированный (многостадийный) пайплайн

См. `combined_pipeline_example.py` — использование `InferencePipeline(graphs=[...])` и `parallel_groups` (REFACTORING_GRAPH_PIPELINE_ENGINE.md §11.7 P1–P4, Фаза 9).

Единый класс без отдельного CombinedPipeline: передаётся один граф, список графов или словарь графов.
