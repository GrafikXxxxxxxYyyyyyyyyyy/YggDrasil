# Фазы оркестратора и роли узлов

Краткий справочник по `GraphBuildOrchestrator` и реестрам. Детали — в [REFACTORING_GRAPH_PIPELINE_ENGINE.md](REFACTORING_GRAPH_PIPELINE_ENGINE.md).

## 1. Фазы сборки

| Фаза | Когда вызывается | Действия |
|------|-------------------|----------|
| **REGISTER** | `add_node(name, type=..., auto_connect=True)` | Определение роли по `block_type` (RoleRegistry), сохранение узла в графе. |
| **RESOLVE** | Сразу после REGISTER | TargetResolver по роли и BuildState возвращает: куда подключить `(graph, node, port)` или **DEFER** (цель ещё не создана). |
| **DEFER_OR_CONNECT** | После RESOLVE | Если цель найдена — добавляется ребро или узел вкладывается (адаптер → inner graph цикла). Если DEFER — запись в `deferred_adapter_bindings`. |
| **MATERIALIZE** | `graph.to(device)` или `orchestrator.materialize()` | Создание циклов из backbone, обработка отложенных адаптеров (вставка в inner graph, проброс входов). |
| **VALIDATE** | После MATERIALIZE или по запросу | Проверка связности, обязательных портов, имён входов. Возвращает `ValidationResult(errors, warnings)`. |

Логика «куда подключить» сосредоточена в **NodeRule** и **resolve_target**; в коде оркестратора нет ветвлений по роли (разд. 10.A спецификации).

## 2. Роли (RoleRegistry)

Маппинг префикса `block_type` → роль и правило подключения.

### Диффузионные

| Роль | block_type (префикс) | insert_into | Примечание |
|------|----------------------|-------------|------------|
| adapter | `adapter/` | loop_inner | ControlNet, IP-Adapter, T2I — в inner graph цикла, порт backbone `adapter_features` (или `image_prompt_embeds`). При отсутствии цикла — DEFER. |
| inner_module | (как adapter) | loop_inner | То же, что adapter. |
| conditioner | `conditioner/` | root | Выход → condition/uncond цикла. |
| backbone | `backbone/` | root | Подмена на LoopSubGraph (creates_loop=True). |
| codec | `codec/` | root | Вход от выхода цикла (latents). |
| solver | `solver/` | loop_inner | Внутри шага цикла. |
| denoise_loop | `loop/` | root | Уже собранный цикл. |

### Недиффузионные (разд. 7, 11.5 N1)

| Роль | block_type (префикс) | insert_into | graph_input |
|------|----------------------|-------------|--------------|
| segmenter | `segmenter/` | root | image |
| detector | `detector/` | root | image |
| classifier | `classifier/` | root | input |
| depth_estimator | `depth_estimator/` | root | image |
| pose_estimator | `pose_estimator/` | root | image |
| super_resolution | `super_resolution/` | root | input |
| style_encoder | `style_encoder/` | root | input |
| feature_extractor | `feature_extractor/` | root | input |
| processor | `processor/` | root | input |

Для недиффузионных графов цикл не создаётся; оркестратор только соединяет узлы по правилам (N3). Контракт входа/выхода — см. `NONDIFFUSION_INPUT_KEYS`, `NONDIFFUSION_OUTPUT_KEYS`, `get_expected_io_for_modality()` в `yggdrasil.core.graph.orchestrator`.

## 3. Реестры и расширяемость

- **LoopTemplates**: `(solver_type, modality)` → шаблон шага цикла (id или callable). Регистрации общие для всех экземпляров (class-level). Регистрация кастомных шаблонов без изменения ядра.
- **StepBuilderRegistry**: `template_id` → builder. При загрузке `image_pipelines` регистрируются **"generic"**, **"step_sdxl"**, **"step_sd3"**, **"step_flux"**. `get_step_template_id_for_metadata(metadata)` по `base_model` возвращает step_sdxl (sdxl/sd15), step_sd3 (sd3), step_flux (flux).
- **SolverRegistry**: имя шеддулера Diffusers → внутренний `solver_type` и опционально `step_signature` (epsilon / v_prediction). Регистрации общие для всех экземпляров (class-level); дефолтный маппинг заполняется при первом создании экземпляра.
- **AdapterBindingRules**: тип адаптера → куда вставить, порт backbone, имя graph_input (control_image, ip_image, …).
- **register_custom_role(block_type_prefix, rule)**: добавление новой роли (например world_model) без правки RoleRegistry.

## 4. Примеры

- **Диффузия:** add_node(conditioner) → root; add_node(backbone) → DEFER, затем при materialize подмена на LoopSubGraph; add_node(adapter) при отсутствии цикла → DEFER, при materialize — вставка в inner graph.
- **Недиффузия:** add_node(segmenter), add_node(detector) — оба в root, граф выполняется без цикла; `pipe(image=...)` без guidance_scale/num_steps (N3).
- **Комбинированный пайплайн:** `InferencePipeline(graphs=[("seg", g_seg), ("gen", g_gen)], parallel_groups=[["seg"], ["gen"]])` — порядок выполнения по уровням (P3).

## 5. Связь с Diffusers (L5)

При импорте пайплайна из Diffusers (`InferencePipeline.from_diffusers(pipe)` или `DiffusersBridge.import_pipeline(pipe)`):

1. Имя класса шеддулера (`type(pipe.scheduler).__name__`) передаётся в **SolverRegistry**: `get_solver_type(scheduler_id)` → внутренний `solver_type` (например `euler_discrete`), опционально `get_step_signature(scheduler_id)` → `prediction_type` (epsilon / v_prediction).
2. В **graph.metadata** записываются `solver_type` и при наличии `prediction_type` для использования шаблонами и циклом.
3. Выбор шаблона шага: **LoopTemplates.get_or_default(solver_type, modality)** — для неизвестной пары возвращается `"generic"`.

Новый шеддулер из Diffusers достаточно зарегистрировать в SolverRegistry (и при необходимости в LoopTemplates), без правки моста.

## 6. Расширяемость (S4)

Добавление новых ролей, адаптеров и шаблонов без изменения ядра:

- **Новая роль (например world_model):** `register_custom_role("world_model/", NodeRule(role="world_model", insert_into="root", ...))`. После этого блоки с `block_type.startswith("world_model/")` обрабатываются по этому правилу.
- **Новый адаптер:** реализовать блок с контрактом адаптера; зарегистрировать в **AdapterBindingRules**: `AdapterBindingRules().register("adapter/my_adapter", AdapterBindingRule(...))` (или расширить _default_adapter_bindings в коде и перезапуск).
- **Новый шаблон шага:** `LoopTemplates().register(solver_type, modality, template_id_or_callable)`. Для неизвестной пары по умолчанию используется `get_or_default(..., "generic")`.
- **Новый step builder (SD3, Flux, …):** зарегистрировать в **StepBuilderRegistry**: `StepBuilderRegistry().register("step_flux", my_flux_step_builder)`; в LoopTemplates зарегистрировать `("flow_match_euler", "image", "step_flux")`; при materialize цикла с таким solver_type/modality будет вызван `my_flux_step_builder(metadata=..., pretrained=..., ...)`.
- **Новый шеддулер:** `SolverRegistry().register("MyScheduler", "my_solver_type", "epsilon")`.

Плагины могут вызывать эти регистрации при импорте модуля (например в `__init__.py` или при первом использовании).
