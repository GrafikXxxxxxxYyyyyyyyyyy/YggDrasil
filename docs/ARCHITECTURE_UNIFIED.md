# Унифицированный пайплайн и Lego-конструктор диффузии

## Идея

Любой диффузионный процесс (изображение, видео, музыка, речь, молекулы, 3D) можно разложить на **одинаковые абстрактные шаги**. Фреймворк задаёт только этот минимум; конкретные архитектуры (UNet, DiT, Encodec, GNN и т.д.) реализуют шаги через единые порты и блоки.

## Канонические шаги (модальность-независимые)

| Шаг | Описание | Типичные блоки |
|-----|----------|----------------|
| **ENCODE** | Сырой сигнал → начальные латенты (или чистый шум) | VAE encoder, Encodec, init noise |
| **CONDITION** | Промпт/класс/другое → эмбеддинг условия | CLIP, T5, CLAP, VL-энкодер |
| **CONTROL** | Управляющий сигнал → adapter_features | ControlNet, T2I-Adapter |
| **REVERSE_STEP** | Один шаг денойзинга: latents, t, condition → next_latents | Backbone + Solver |
| **DENOISE_LOOP** | Цикл REVERSE_STEP по таймстепам | LoopSubGraph |
| **DECODE** | Латенты → выходной сигнал | VAE decoder, Encodec |
| **POST_PROCESS** | Улучшение результата | FaceDetailer, upscaler |

Обучение добавляет шаги: **FORWARD_PROCESS**, **LOSS**.

## Модальности

- **Image** — pixel ↔ latent (VAE), условие текст/класс, контроль: ControlNet, T2I.
- **Video** — то же + временная размерность; контроль по кадрам.
- **Audio / Speech / Music** — waveform ↔ latent (Encodec и др.), условие текст/метаданные.
- **Molecular** — граф/латент; условие свойства, контроль по скелету.
- **3D** — point cloud / latent; условие текст, контроль по виду.

Тип сигнала задаётся в `TensorSpec.modality` и в `yggdrasil.core.unified.Modality`.

## Единый контракт пайплайна

Граф экспортирует **входы** и **выходы** по каноническим именам (где применимо):

- Входы: `prompt`, `negative_prompt`, `condition`, `control_image`, `control_signal`, `source_image`, `strength`, `initial_latents`, `timesteps`, `guidance_scale`, …
- Выходы: `latents`, `images`, `output`, `output_signal`, `enhanced`.

По графу строится **PipelineContract** (`infer_contract(graph)`): список входов/выходов и соответствие узлов шагам. На этом строится динамический UI (Gradio) и API.

## Lego-принцип

1. **Блок** — минимальная единица: `declare_io()` + `process(**port_inputs)`.
2. **Граф** — DAG из блоков и рёбер по портам.
3. **Шаблон** — функция, возвращающая граф (txt2img, img2img, controlnet, video, training, …).
4. **Адаптеры** — подключаются к backbone через слот `adapter_features` (ControlNet, LoRA, T2I-Adapter и т.д.).
5. **Пост-процессоры** — отдельный блок после декодера: вход/выход — один тип сигнала.

Любая **кастомная модель** — это новый блок (backbone/codec/conditioner), реализующий те же порты. Архитектура может быть произвольной (UNet, DiT, Transformer, GNN).

## Обучение

- **Полная модель** — граф с шагами CONDITION → backbone → LOSS; обучаются параметры backbone (и при необходимости conditioner).
- **Адаптеры** — тот же граф, обучаются только узлы-адаптеры (LoRA, Textual Inversion, ControlNet и т.д.); остальное заморожено.
- Шаблоны обучения в `training_pipelines.py`; один и тот же граф можно использовать для inference (без LOSS) и для training (с LOSS и датасетом).

## Подключение ControlNet и аналогов

- **Как отдельный блок**: в граф добавляется узел ControlNet, его выход `output` (down/mid residuals) подключается к backbone в порт `adapter_features`. Вход графа `control_image` соединяется с этим узлом.
- Функция `add_controlnet_to_graph(graph, ...)` и метод `graph.with_controlnet(...)` делают это автоматически для графов с подходящим backbone (не batched CFG без адаптеров).
- Аналогично подключаются другие адаптеры (T2I-Adapter, кастомные), если backbone принимает `adapter_features`.

## Gradio

- Текущий UI привязан к конкретным полям (prompt, steps, …).
- **Расширение**: по загруженному графу вызывать `infer_contract(graph)` и строить форму по `contract.get_input_names()` и типам портов; вывод — по `contract.get_output_names()`. Тогда один интерфейс обслуживает любой пайплайн (изображение, видео, аудио, с/без контроля, с пост-процессором).
- В `yggdrasil.serving.contract_bridge` есть `get_pipeline_contract(graph)` и `contract_to_ui_hints(contract)` для подготовки подсказок под динамический UI.

## Файлы

- **Ядро унификации**: `yggdrasil/core/unified/` — `Modality`, `DiffusionStep`, `PipelineContract`, `infer_contract`.
- **Абстракции моделей**: `yggdrasil/core/model/` — Backbone, Codec, Conditioner, Guidance, **PostProcessor**.
- **Граф**: `yggdrasil/core/graph/` — ComputeGraph, шаблоны, `add_controlnet_to_graph`.
- **Сборка блоков**: `yggdrasil/core/block/` — AbstractBlock, Port, Registry, Builder.
