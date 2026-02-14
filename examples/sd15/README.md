# Stable Diffusion 1.5 — примеры и гайд по YggDrasil

Этот каталог содержит **полный набор примеров** использования фреймворка YggDrasil на базе Stable Diffusion 1.5: от базовой генерации до кастомных пайплайнов, замены компонентов и сборки графа из блоков.

---

## Возможности фреймворка (на примере SD 1.5)

| Возможность | Описание |
|-------------|----------|
| **Готовые шаблоны** | `sd15_txt2img`, `sd15_img2img`, `sd15_inpainting` — один вызов `from_template()` |
| **Замена компонентов** | Замена solver, guidance, conditioner, backbone через `graph.replace_node()` без пересборки графа |
| **Кастомные параметры** | `guidance_scale`, `num_steps`, `negative_prompt`, `seed`, `width`, `height` — передаются в `pipe()` или в `graph.execute()` |
| **Сборка из блоков** | Ручная сборка графа из блоков (BlockBuilder + ComputeGraph) для своих пайплайнов |
| **Прямое выполнение графа** | Вызов `graph.execute(**inputs)` без Pipeline — полный доступ к входам/выходам |
| **Загрузка по HF ID** | `Pipeline.from_pretrained("runwayml/stable-diffusion-v1-5")` — автоматический выбор шаблона |
| **ControlNet и адаптеры** | Шаблон `controlnet_txt2img` и инъекция LoRA/ControlNet в существующий граф |

---

## Структура примеров

| Файл | Назначение | Ключевые возможности |
|------|------------|----------------------|
| [generate.py](#generatepy) | Базовая текстовая генерация | Шаблон `sd15_txt2img`, Pipeline API |
| [generate_diffusers.py](#generate_diffuserspy) | Эталон через HuggingFace Diffusers | Сравнение качества с YggDrasil |
| [02_from_pretrained.py](#02_from_pretrainedpy) | Загрузка по имени модели / HF ID | `Pipeline.from_pretrained()` |
| [03_img2img.py](#03_img2imgpy) | Image-to-image | Шаблон `sd15_img2img`, вход `source_image` |
| [04_custom_parameters.py](#04_custom_parameterspy) | Кастомные параметры генерации | `guidance_scale`, `num_steps`, `negative_prompt` |
| [05_replace_solver.py](#05_replace_solverpy) | Замена шага диффузии (solver) | `graph.replace_node("solver", ...)` |
| [06_replace_guidance.py](#06_replace_guidancepy) | Замена блока CFG | Другой `guidance_scale` или отключение CFG |
| [07_build_minimal_graph.py](#07_build_minimal_graphpy) | Сборка графа из блоков вручную | BlockBuilder, ComputeGraph, LoopSubGraph |
| [08_execute_graph_directly.py](#08_execute_graph_directlypy) | Выполнение без Pipeline | `graph.execute()`, работа с сырыми входами/выходами |
| [09_controlnet.py](#09_controlnetpy) | ControlNet (Canny и др.) | Шаблон `controlnet_txt2img` |
| [10_list_templates_blocks.py](#10_list_templates_blockspy) | Список шаблонов и блоков | API для обзора возможностей |

---

## Подробное описание файлов

### generate.py

**Что делает:** Загружает пайплайн из шаблона `sd15_txt2img` и генерирует одно изображение по текстовому промпту. Сохраняет результат в `output.png` и выводит диагностику по устройствам и dtype блоков.

**Как запустить:**
```bash
cd /path/to/YggDrasil
python examples/sd15/generate.py
```

**Возможности для разработки:**
- Стартовая точка для любого пайплайна SD 1.5.
- Доступ к графу: `pipe.graph`, итерация по блокам: `pipe.graph._iter_all_blocks()`.

---

### generate_diffusers.py

**Что делает:** Генерирует изображение тем же промптом и seed через официальный пайплайн HuggingFace Diffusers. Сохраняет в `output_diffusers.png` для сравнения с `output.png` от YggDrasil.

**Как запустить:**
```bash
python examples/sd15/generate_diffusers.py
```

**Возможности для разработки:**
- Эталон качества и воспроизводимости при отладке своего пайплайна.

---

### 02_from_pretrained.py

**Что делает:** Загружает пайплайн по имени пресета (`"sd15"`) или по HuggingFace model ID (`"runwayml/stable-diffusion-v1-5"`). Показывает, что один и тот же код может работать с разными способами указания модели.

**Как запустить:**
```bash
python examples/sd15/02_from_pretrained.py
```

**Возможности для разработки:**
- Единая точка входа для своих пресетов и HF-моделей через `from_pretrained()`.

---

### 03_img2img.py

**Что делает:** Использует шаблон `sd15_img2img`, который добавляет вход `source_image`. Пример передаёт изображение по пути и генерирует вариацию (при наличии правильной подготовки латентов в графе).

**Как запустить:**
```bash
python examples/sd15/03_img2img.py
# При необходимости указать путь к изображению через аргумент.
```

**Возможности для разработки:**
- Расширение графа дополнительными входами (`expose_input`) и использование одного графа для txt2img/img2img.

---

### 04_custom_parameters.py

**Что делает:** Генерирует изображение с явной передачей `guidance_scale`, `num_steps`, `negative_prompt`, `seed`, `width`, `height`. Демонстрирует, что параметры можно менять без пересборки графа.

**Как запустить:**
```bash
python examples/sd15/04_custom_parameters.py
```

**Возможности для разработки:**
- Проверка влияния параметров на результат; передача своих kwargs в `pipe()` и дальше в `graph.execute()`.

---

### 05_replace_solver.py

**Что делает:** Берёт граф из шаблона `sd15_txt2img`, находит узел solver во внутреннем графе цикла деноайзинга и заменяет его на новый экземпляр DDIM-решателя с другими параметрами (например, `eta`). Генерация идёт уже с новым solver’ом.

**Как запустить:**
```bash
python examples/sd15/05_replace_solver.py
```

**Возможности для разработки:**
- Подмена любого solver’а (другой тип или конфиг) без изменения остального графа; паттерн `replace_node` для экспериментов.

---

### 06_replace_guidance.py

**Что делает:** Заменяет блок guidance (CFG) в шаге деноайзинга на новый с другим `scale` или отключает CFG (scale ≤ 1). Показывает точечную замену компонента, отвечающего за силу текстовой conditioning.

**Как запустить:**
```bash
python examples/sd15/06_replace_guidance.py
```

**Возможности для разработки:**
- A/B-тесты по силе guidance; подключение своих блоков guidance с той же контрактной обвязкой портов.

---

### 07_build_minimal_graph.py

**Что делает:** Собирает минимальный текст-к-картинке граф «с нуля»: создаёт блоки через `BlockBuilder.build(config)`, собирает шаг деноайзинга (backbone → guidance → solver), оборачивает в `LoopSubGraph`, подключает conditioner и codec. Не использует готовый шаблон.

**Как запустить:**
```bash
python examples/sd15/07_build_minimal_graph.py
```

**Возможности для разработки:**
- Понимание структуры графа SD 1.5; основа для своих архитектур (другие backbone/conditioner/solver/codec).

---

### 08_execute_graph_directly.py

**Что делает:** Создаёт граф из шаблона и вызывает `graph.execute(prompt=..., latents=..., negative_prompt=..., ...)` напрямую, без Pipeline. Получает сырой словарь выходов (`decoded`, `latents` и т.д.) и сохраняет изображение вручную.

**Как запустить:**
```bash
python examples/sd15/08_execute_graph_directly.py
```

**Возможности для разработки:**
- Полный контроль над входами (в т.ч. свои латенты, эмбеддинги); интеграция графа в свои скрипты без обёртки Pipeline.

---

### 09_controlnet.py

**Что делает:** Загружает пайплайн ControlNet для SD 1.5 (шаблон `controlnet_txt2img`), передаёт текстовый промпт и контрольное изображение (например, Canny), генерирует изображение и сохраняет результат.

**Как запустить:**
```bash
python examples/sd15/09_controlnet.py
# При необходимости указать путь к контрольному изображению.
```

**Возможности для разработки:**
- Добавление пространственного контроля (Canny, Depth, OpenPose и т.д.) поверх базового SD 1.5 через один шаблон.

---

### 10_list_templates_blocks.py

**Что делает:** Выводит список зарегистрированных шаблонов графов и список зарегистрированных типов блоков (backbone, guidance, conditioner, codec, solver, adapter и т.д.). Помогает понять, какие «кирпичики» доступны для сборки и замены.

**Как запустить:**
```bash
python examples/sd15/10_list_templates_blocks.py
```

**Возможности для разработки:**
- Обзор API фреймворка; выбор типа блока для `BlockBuilder.build({"type": "..."})` и для своих шаблонов.

---

## Общие требования

- Python 3.10+
- Установленный пакет `yggdrasil` и зависимости (PyTorch, diffusers, transformers и т.д.)
- Для генерации изображений: ~5 GB места под веса SD 1.5 (скачиваются с HuggingFace при первом запуске)
- Устройство: CUDA, MPS (Apple Silicon) или CPU

Запуск из корня репозитория (чтобы в пути был YggDrasil):

```bash
cd /path/to/YggDrasil
export PYTHONPATH=.   # при необходимости
python examples/sd15/<script>.py
```

---

## Связь с документацией фреймворка

- **Шаблоны графов:** `yggdrasil/core/graph/templates/image_pipelines.py` (sd15_txt2img, sd15_img2img, sd15_inpainting).
- **Блоки:** `yggdrasil/blocks/` (backbones, conditioners, guidances, codecs, adapters).
- **Сборка блоков:** `yggdrasil/core/block/builder.py` (BlockBuilder.build).
- **Граф и выполнение:** `yggdrasil/core/graph/graph.py` (ComputeGraph), `yggdrasil/core/graph/executor.py`.
- **Pipeline API:** `yggdrasil/pipeline.py`.

---

## Возможности для разработки (кратко)

- **Свои блоки:** реализовать класс с `declare_io()` и `process()`, зарегистрировать через `@register_block("type/name")`, собирать граф через `BlockBuilder.build({"type": "type/name", ...})`.
- **Свои шаблоны:** функция, возвращающая `ComputeGraph`, декоратор `@register_template("name")` — затем `ComputeGraph.from_template("name")`.
- **Замена узлов:** `graph.replace_node("node_name", new_block)` — проверка портов опциональна (`validate=False`).
- **Вложенные графы:** `LoopSubGraph.create(inner_graph=step_graph, ...)` — цикл как один блок с теми же входами/выходами, что и внутренний граф.
- **Прямое выполнение:** `GraphExecutor(no_grad=True).execute(graph, **inputs)` — без Pipeline, полный контроль над входами и сырыми выходами.

Эти примеры покрывают все перечисленные возможности на базе Stable Diffusion 1.5 и служат полным гайдом по использованию и расширению фреймворка.
