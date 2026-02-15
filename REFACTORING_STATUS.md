# Статус рефакторинга YggDrasil (ТЗ v1.2)

Рефакторинг по TECHNICAL_SPECIFICATION.md v1.2 считается **закрытым**: все критерии приёмки (раздел 9) и этапы реализации (раздел 8) выполнены.

---

## Критерии приёмки (раздел 9 ТЗ)

| № | Критерий | Статус |
|---|----------|--------|
| 1 | Все блоки в коде наследуются от AbstractBaseBlock | ✅ Все зарегистрированные блоки (register_block) наследуют AbstractBaseBlock или его потомков (AbstractBackbone, AbstractAdapter, AbstractSolver и т.д.) |
| 2 | Реализованы абстракции: Backbone, Codec, Conditioner, Guidance, Solver, Adapter, InnerModule, OuterModule, **Processor**; многостадийный пайплайн как цепочка **AbstractStage** | ✅ Все перечислены в core/model, blocks, core/graph/stage.py; Processor — processor/abstract в core/model/processor.py; составной пайплайн = граф AbstractStage |
| 3 | В графе нет отдельного типа «Scheduler» — только Solver | ✅ Параметры расписания в solver_config; schedule/ маппится на solver в role_rules |
| 4 | **add_node(type="...")** с автоопределением роли и подключения; **add_stage** с самоопределением последовательности по портам; Detailer/Upscaler — стадии | ✅ add_node(type=..., auto_connect=True) + role_rules (backbone, codec, conditioner, solver, adapter→backbone.adapter_features, inner_module, outer_module, processor); add_stage(..., auto_connect_by_ports=True); Detailer/Upscaler как AbstractStage в шаблонах |
| 5 | Любой диффузионный пайплайн представляется графом из компонентов | ✅ ComputeGraph + шаблоны для всех семейств моделей |
| 6 | InferencePipeline и TrainingPipeline: from_config, from_pretrained, from_graph, from_template, простой API | ✅ Реализованы в yggdrasil/pipeline.py; from_diffusers, from_combined добавлены |
| 7 | Поддержка обучения любых блоков, выбор обучаемых узлов, multi-LoRA | ✅ GraphTrainer(train_nodes=...), TrainingPipeline(train_nodes=..., train_stages=...); load_lora_weights в integration/lora_loader.py |
| 8 | Шаблоны/тесты для SD 1.5, SDXL, FLUX.1, FLUX.2 Klein, SD3 | ✅ Шаблоны в core/graph/templates; tests/test_model_families.py, test_lego_constructor.py |
| 9 | Поддержка Diffusers и деплоя на удалённые платформы | ✅ DiffusersBridge, InferencePipeline.from_diffusers; Modal, RunPod, Vast.ai с единым интерфейсом (InferencePipeline / default_graph) |
| 10 | Комбинирование графов: составной пайплайн через конфиг, один вызов InferencePipeline | ✅ kind: combined_pipeline, from_combined, from_workflow, Runner.execute |
| 11 | Обучение комбинированного пайплайна через TrainingPipeline: train_stages/train_nodes, чекпоинты | ✅ train_stages, train_nodes с префиксом стадии (stage0/...); GraphTrainer.save_checkpoint, TrainingPipeline.save_checkpoint |
| 12 | Универсальность: любые модальности, регистрация своей модальности/модели/пайплайна, обучение любого блока | ✅ Плагины (plugins), регистрация блоков и шаблонов без изменения ядра |

---

## Этапы реализации (фазы 1–5)

- **Фаза 1:** AbstractBaseBlock, абстракции уровня 1, AbstractStage, Scheduler объединён с Solver ✅  
- **Фаза 2:** add_node(type=..., auto_connect), правила подключения по ролям (role_rules.py), YAML/JSON графа и составного пайплайна ✅  
- **Фаза 3:** InferencePipeline, TrainingPipeline, многостадийный пайплайн, from_config/from_pretrained/from_graph/from_workflow/from_combined ✅  
- **Фаза 4:** SD 1.5, SDXL, FLUX.1, FLUX.2, SD3 (шаблоны, адаптеры, LoRA, обучение) ✅  
- **Фаза 5:** Стабилизация Diffusers (единый Solver, from_diffusers, плоский граф); деплой (Modal, RunPod, Vast.ai) под единый интерфейс ✅  

---

## Основные файлы

- **Граф:** `yggdrasil/core/graph/graph.py` (ComputeGraph, add_node, add_stage, from_yaml, to_yaml, from_workflow, to_workflow, combined_pipeline)
- **Роли:** `yggdrasil/core/graph/role_rules.py` (TYPE_TO_ROLE, get_connection_rules)
- **Стадия:** `yggdrasil/core/graph/stage.py` (AbstractStage)
- **Пайплайны:** `yggdrasil/pipeline.py` (InferencePipeline, TrainingPipeline)
- **Обучение:** `yggdrasil/training/graph_trainer.py` (train_nodes, train_stages, save_checkpoint)
- **Шаблоны:** `yggdrasil/core/graph/templates/` (image, control, video, audio, training)
- **Деплой:** `yggdrasil/deployment/cloud/` (modal_app, runpod, vastai)
- **Интеграция:** `yggdrasil/integration/diffusers.py` (DiffusersBridge, import_pipeline)

Рефакторинг закрыт. Дальнейшие доработки — по новым требованиям или поддержке дополнительных сценариев.
