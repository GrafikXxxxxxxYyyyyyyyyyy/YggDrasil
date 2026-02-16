# Примеры YggDrasil

- **[images/](images/)** — генерация изображений: сравнение с Diffusers (SD 1.5, SDXL, SD3), гайды.
- **[pipelines/](pipelines/)** — комбинированные (многостадийные) пайплайны: `InferencePipeline(graphs=[...])`, `parallel_groups`, `from_spec`. См. [REFACTORING_GRAPH_PIPELINE_ENGINE.md](../docs/REFACTORING_GRAPH_PIPELINE_ENGINE.md) §11.7, Фаза 9.

**Gradio UI** (`yggdrasil ui`): 5 вкладок (Inference, Pipeline, Train, Blocks, Philosophy). На Inference: кнопка «Загрузить и показать входы» — отображает graph_inputs графа; поле «Доп. параметры (JSON)» — передача любых входов, не охваченных формой. Materialize на Pipeline обновляет список входов на Inference.

Запуск примеров из корня репозитория: `PYTHONPATH=. python examples/...`
