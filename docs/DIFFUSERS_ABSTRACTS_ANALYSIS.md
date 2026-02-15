# Анализ: достаточность набора абстрактных блоков относительно Diffusers

**Цель:** Проверить по репозиторию Hugging Face Diffusers (v0.36.0), является ли текущий набор абстракций в ТЗ исчерпывающим для любой диффузионной задачи. Если нет — перечислить, какие абстракции имеет смысл добавить (в ТЗ пока не вносим).

---

## 1. Что есть в Diffusers (сводка по разделам)

### 1.1 Модели (`models/`)

- **Трансформеры / UNet:** UNet2DConditionModel, FluxTransformer2DModel, SD3Transformer2DModel, WanTransformer3DModel, DiT, CogVideoX, HunyuanVideo, Latte, StableAudioDiTModel, и др. → у нас покрываются **AbstractBackbone**.
- **Автоэнкодеры (VAE):** AutoencoderKL, AutoencoderKLFlux2, AutoencoderKLWan, VQModel, ConsistencyDecoderVAE и др. → у нас **AbstractCodec**.
- **ControlNet / T2I / адаптеры:** ControlNetModel, T2IAdapter, MultiControlNetModel, FluxControlNetModel, SD3ControlNetModel, SparseControlNetModel и др. → у нас **AbstractInnerModule** (и при необходимости AbstractAdapter для LoRA-подобных).
- **PriorTransformer:** используется в Kandinsky, Wuerstchen, Stable Cascade — отдельная **генеративная** стадия (текст → латент/эмбеддинг для декодера). В Diffusers это отдельный компонент пайплайна, не «conditioner» и не «backbone» основной диффузии.
- **MotionAdapter:** добавляет временное измерение к 2D UNet (AnimateDiff). Семантически — «временной/мotion-модуль», встраиваемый в процесс генерации.
- **MultiAdapter, IP-Adapter:** загрузка и применение нескольких адаптеров → у нас AbstractAdapter + multi-LoRA.
- **Projection-модели:** CLIPImageProjection, AudioLDM2ProjectionModel, ReduxImageEncoder — проекция из одного эмбеддинг-пространства в другое (например, изображение → CLIP-пространство для Redux).

### 1.2 Планировщики и шаги (`schedulers/`)

- Много реализаций: DDIM, Euler, DPM, PNDM, FlowMatchEuler, KarrasVe, и т.д. → у нас все уходят в **AbstractSolver** (Scheduler объединён с Solver в ТЗ). Исчерпывающе.

### 1.3 Guiders (`guiders/`)

- ClassifierFreeGuidance, PerturbedAttentionGuidance (PAG), SkipLayerGuidance, AdaptiveProjectedGuidance, FrequencyDecoupledGuidance, SmoothedEnergyGuidance, TangentialClassifierFreeGuidance и др. → у нас **AbstractGuidance**. Исчерпывающе.

### 1.4 Hooks (`hooks/`)

- Не «блоки» в смысле графа, а модификаторы инференса: кэши (FasterCache, FirstBlockCache, TaylorSeerCache), LayerSkip, PyramidAttentionBroadcast, group offloading, layerwise casting. В YggDrasil уже есть pre_hooks/post_hooks на блоках; при необходимости такие вещи можно реализовать как хуки или как отдельный слой «оптимизации инференса» без обязательной новой абстракции блока.

### 1.5 Процессоры и утилиты

- **image_processor.py, video_processor.py:** предобработка изображений/видео (resize, crop, normalize) перед подачей в модель. В Diffusers это отдельные объекты, не «conditioner» и не «inner module».
- **quantizers:** BitsAndBytes, GGUF, Quanto и т.д. — квантизация весов (деплой/оптимизация), а не вычислительный блок графа.

### 1.6 Loaders

- Загрузка LoRA, IP-Adapter, Textual Inversion, single-file моделей и т.д. — это механизмы загрузки/применения, а не новые роли блоков в графе. Остаются реализациями существующих абстракций (Adapter, Conditioner и т.д.).

### 1.7 Modular pipelines

- ModularPipeline, ComponentSpec, AutoBlocks — способ сборки пайплайна из блоков и конфигов. По смыслу близко к нашему ComputeGraph + шаблоны; не вводят новых *типов* блоков.

### 1.8 Двухстадийные пайплайны

- **Kandinsky:** Prior (текст → эмбеддинг/латент) + основной диффузионный декодер.
- **Wuerstchen / Stable Cascade:** Prior + Decoder (две последовательные диффузионные или генеративные стадии).
- **Stable Cascade:** явно Prior + Decoder.

Общая суть: есть **отдельная генеративная стадия «Prior»**, результат которой является входом (условием или латентом) для следующей стадии. Это не Conditioner (кодирует внешний сигнал) и не Backbone основной диффузии — это **отдельная роль**: «генератор условия/латента для следующей стадии».

---

## 2. Текущий набор абстракций в ТЗ (напоминание)

- AbstractBaseBlock  
- AbstractBackbone  
- AbstractCodec  
- AbstractConditioner  
- AbstractGuidance  
- AbstractSolver (включая бывший Scheduler)  
- AbstractAdapter  
- AbstractInnerModule  
- AbstractOuterModule  

Плюс отдельно: AbstractDiffusionProcess (математика процесса).

---

## 3. Вывод: чего не хватает для исчерпывающего покрытия

С учётом Diffusers и типичных задач (изображение, видео, аудио, двухстадийные и составные пайплайны) ниже перечислены **кандидаты в новые абстракции**. Имеет смысл обсудить их и при необходимости добавить в ТЗ; здесь они **в ТЗ не вносятся**, только фиксируются.

### 3.1 AbstractPrior (или эквивалент по смыслу)

- **Зачем:** Двухстадийные пайплайны (Kandinsky, Wuerstchen, Stable Cascade): первая стадия — «Prior» — генерирует латент или эмбеддинг, вторая стадия — основная диффузия — использует это как условие или начальный латент.
- **Отличие от Conditioner:** Conditioner кодирует *внешний* вход (текст, изображение и т.д.). Prior — это *генеративная* модель (диффузия или иная), выход которой подаётся в следующую стадию.
- **Отличие от Backbone:** Backbone — ядро одной диффузионной стадии. Prior — целая предыдущая стадия (со своим процессом/сольвером), результат которой — условие/латент для следующего графа.
- **Варианты:** ввести явную абстракцию **AbstractPrior** (блок «prior-стадии») или описать двухстадийность только композицией графов (первый граф — «prior», второй — «decoder»), без отдельного типа блока. Для единообразия и явной поддержки Kandinsky/Cascade/Wuerstchen удобно иметь **AbstractPrior** как блок, который может быть целым подграфом или одной моделью.

**Рекомендация:** Добавить **AbstractPrior** в список абстрактных ролей, если хотим явно и единообразно поддерживать любые двухстадийные (prior + decoder) пайплайны.

---

### 3.2 AbstractPreprocessor

- **Зачем:** В Diffusers есть `image_processor`, `video_processor` — детерминированная предобработка (resize, crop, normalize, to tensor) перед подачей в модель. Сейчас в YggDrasil такую логику часто зашивают в блоки (например, внутрь ControlNet или conditioner). Для явного графа и повторного использования удобно вынести «сырой вход → нормализованный тензор» в отдельный тип узла.
- **Роль:** Вход: сырые данные (PIL, путь к файлу, numpy, видео). Выход: тензор(ы), готовые к подаче в следующий блок (например, control_image, ip_image).
- **Варианты:** ввести **AbstractPreprocessor** (или AbstractInputProcessor) для любых детерминированных преобразований входа; либо считать это частью Conditioner/InnerModule и не выделять. Для «любой диффузионной задачи» и чистоты графа лучше иметь явный тип узла.

**Рекомендация:** Рассмотреть добавление **AbstractPreprocessor** (или аналогичного имени) для явной предобработки изображений/видео/аудио в графе.

---

### 3.3 AbstractMotionModule / AbstractTemporalAdapter

- **Зачем:** В Diffusers **MotionAdapter** (AnimateDiff) добавляет временное измерение 2D UNet’у. Семантически это не «просто» ControlNet и не классический LoRA — это «модуль, вводящий временную согласованность».
- **Варианты:** трактовать как частный случай **AbstractInnerModule** или **AbstractAdapter** (модуль, встраиваемый в backbone и меняющий поведение по времени). Если хотим явно отразить в онтологии «мotion/temporal» для видео и анимации, можно ввести **AbstractMotionModule** или **AbstractTemporalAdapter**; иначе оставляем покрытие через Inner/Adapter.

**Рекомендация:** Пока можно не вводить отдельную абстракцию и считать MotionAdapter реализацией **AbstractInnerModule** (или Adapter). Если позже появятся и другие «временные» модули с иным контрактом — тогда ввести **AbstractTemporalAdapter** / **AbstractMotionModule**.

---

### 3.4 AbstractProjection

- **Зачем:** В Diffusers есть проекции: CLIPImageProjection, AudioLDM2ProjectionModel, ReduxImageEncoder — отображение из одного эмбеддинг-пространства в другое (например, под размер cross-attention основной модели). По смыслу это «эмбеддинг → эмбеддинг», не полноценный Conditioner (который часто «сырой сигнал → эмбеддинг»).
- **Варианты:** считать проекции разновидностью **AbstractConditioner** (один вход — эмбеддинг, выход — эмбеддинг) или ввести **AbstractProjection** для явного отличия «encoder условия» от «проекции между пространствами».

**Рекомендация:** Опционально. Можно оставить в рамках Conditioner; при желании явно разделить «кодировщик условия» и «проекция эмбеддингов» — добавить **AbstractProjection**.

---

### 3.5 Quantization / Inference optimization

- **В Diffusers:** quantizers (BitsAndBytes, GGUF, …), hooks (cache, layer skip, offload). Это не «вычислительные блоки» графа, а обёртки/оптимизации.
- **В YggDrasil:** можно оставить как утилиты, конфигурацию деплоя или обёртки над блоками без нового типа Abstract* в ядре графа.

**Рекомендация:** Не добавлять новые абстрактные блоки; достаточно хуков и конфигурации квантизации/оффлоада.

---

## 4. Итоговая таблица: что добавить (только предложения, в ТЗ не вносилось)

| Кандидат              | Назначение                                                                 | Приоритет для «любой диффузионной задачи» |
|-----------------------|----------------------------------------------------------------------------|-------------------------------------------|
| **AbstractPrior**     | Двухстадийные пайплайны (Kandinsky, Wuerstchen, Stable Cascade): prior-стадия как отдельная роль. | **Высокий** — без этого двухстадийность выражается только композицией графов без явной роли «prior». |
| **AbstractPreprocessor** | Явная предобработка входов (image/video/audio processor) в графе.       | **Средний** — улучшает ясность и переиспользование; можно временно зашить в другие блоки. |
| **AbstractMotionModule / AbstractTemporalAdapter** | Явная роль «motion/temporal» (AnimateDiff и аналоги).              | **Низкий** — можно покрыть AbstractInnerModule/Adapter. |
| **AbstractProjection** | Проекция между эмбеддинг-пространствами (CLIPImageProjection и т.п.).   | **Низкий** — можно покрыть AbstractConditioner. |

---

## 5. Краткий вывод

- Текущего набора абстракций **достаточно** для большинства «одностадийных» диффузионных пайплайнов (SD, FLUX, видео как одна стадия и т.д.).
- Для **исчерпывающего** покрытия любых диффузионных задач, включая двухстадийные и составные пайплайны из Diffusers, целесообразно **добавить в ТЗ одну явную абстракцию: AbstractPrior** (или эквивалент по смыслу «генеративная prior-стадия»).
- **AbstractPreprocessor** имеет смысл рассмотреть для явной и единообразной предобработки входов в графе.
- **AbstractMotionModule** и **AbstractProjection** при необходимости можно ввести позже; текущие роли (Inner, Adapter, Conditioner) их частично покрывают.

В ТЗ на данном этапе ничего не добавлялось — только зафиксирован этот анализ для последующего решения.
