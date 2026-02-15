# YggDrasil

Универсальный Lego-фреймворк для диффузионных моделей: изображения, видео, аудио — любые модальности и любые модели.

## Установка

**Из PyPI (после публикации):**
```bash
pip install yggdrasil
```

**Из исходников (клонированный репозиторий):**
```bash
cd YggDrasil
pip install .
```

**Режим разработки (editable):**
```bash
pip install -e .
```

**Из Git:**
```bash
pip install git+https://github.com/your-org/YggDrasil.git
```

**Публикация на PyPI** (чтобы работало `pip install yggdrasil` для всех):
```bash
pip install build twine
python -m build
twine upload dist/*
# или для Test PyPI: twine upload --repository testpypi dist/*
```

После установки доступна команда `yggdrasil`:
```bash
yggdrasil ui          # Gradio-интерфейс
yggdrasil ui --share  # с публичной ссылкой
yggdrasil api         # REST API
python -m yggdrasil ui  # то же через модуль
```

Дополнительные зависимости (обучение, LoRA, графы):
```bash
pip install yggdrasil[full]   # networkx, safetensors, peft, torchvision, ...
pip install yggdrasil[train]  # для обучения
pip install yggdrasil[dev]    # pytest, ruff, mypy
```

---

## Структура проекта

YggDrasil/
│
├── core/                          # 🔨 ФУНДАМЕНТ LEGO (только интерфейсы и инструменты сборки)
│   ├── __init__.py
│   │
│   ├── block/                     # ★ ВСЁ — ЭТО BLOCK
│   │   ├── base.py                # AbstractBaseBlock (id, slots, config, build(), forward_hook)
│   │   ├── registry.py            # @register_block("category/name") + auto-discovery
│   │   ├── builder.py             # BlockBuilder.build(config) → рекурсивная сборка графа
│   │   ├── slot.py                # Slot (имя, тип, multiple=True, optional=True)
│   │   └── graph.py               # BlockGraph (визуализация, валидация зависимостей)
│   │
│   ├── diffusion/                 # Математические Lego-блоки процесса
│   │   ├── process.py             # AbstractDiffusionProcess (forward, reverse_step)
│   │   ├── noise/
│   │   │   ├── schedule.py        # NoiseSchedule (linear, cosine, sigmoid, custom)
│   │   │   └── sampler.py         # NoiseSampler
│   │   ├── solver/
│   │   │   ├── base.py
│   │   │   ├── ddim.py
│   │   │   ├── heun.py
│   │   │   └── custom_ode.py
│   │   ├── flow.py                # RectifiedFlow, EDM, OptimalTransport
│   │   └── consistency.py
│   │
│   ├── model/                     # Блоки модели (все подключаются через slots)
│   │   ├── modular.py             # ModularDiffusionModel (единственная модель!)
│   │   ├── backbone.py            # AbstractBackbone (любой Transformer/UNet/DiT)
│   │   ├── codec.py               # AbstractLatentCodec
│   │   ├── conditioner.py         # AbstractConditioner (text, image, control, multi)
│   │   ├── guidance.py            # AbstractGuidance (CFG, PAG, FreeU, custom)
│   │   └── position.py            # AbstractPositionEmbedder (n-dimensional)
│   │
│   ├── engine/                    # Движок сборки и выполнения
│   │   ├── sampler.py             # DiffusionSampler (process + solver + guidance)
│   │   ├── pipeline.py            # AbstractPipeline (train_step, infer_step, save)
│   │   ├── state.py               # DiffusionState (latents, t, condition, cache)
│   │   └── loop.py                # SamplingLoop (с хуками на каждый шаг)
│   │
│   └── utils/
│       ├── tensor.py              # DiffusionTensor (any dim + metadata)
│       ├── config.py              # OmegaConf + inheritance + validation + slots
│       └── hooks.py               # Pre/post hooks для любого блока
│
├── blocks/                        # ★ КОНКРЕТНЫЕ LEGO-КИРПИЧИКИ (авторегистрация)
│   ├── diffusion/                 # 20+ процессов
│   │   ├── ddpm.py
│   │   ├── flow_matching.py
│   │   ├── consistency_distillation.py
│   │   └── ...
│   ├── backbones/                 # UNet2D, DiT, MMDiT, 1D-Transformer, EquivariantGNN...
│   ├── codecs/                    # VAE, VQGAN, Encodec, GaussianSplattingCodec...
│   ├── conditioners/              # CLIP, T5, CLAP, ControlNetEmbedder, MultiModal...
│   ├── guidances/                 # CFG, PAG, AttentionControl, SpatialGuidance...
│   ├── adapters/                  # LoRA, ControlNet, IP-Adapter, DoRA, HyperNetwork...
│   └── noise/                     # PerlinNoise, FractalNoise, LowDiscrepancy...
│
├── plugins/                       # ★ ПЛАГИНЫ (модальности + кастомные наборы)
│   ├── __init__.py                # auto-load всех плагинов
│   ├── base.py                    # AbstractPlugin (register_blocks(), default_config)
│   │
│   ├── image/                     # Изображения (SDXL, Flux, SD3, Lumina...)
│   ├── video/                     # Видео (CogVideoX, Hunyuan, Mochi...)
│   ├── audio/                     # Аудио (AudioLDM, StableAudio, MusicGen...)
│   ├── 3d/                        # 3D (Gaussian Splatting, Mesh, PointCloud...)
│   ├── molecular/                 # Молекулы (DiffDock, EquiFold, GeoLDM...)
│   ├── timeseries/                # Временные ряды (новая!)
│   ├── text/                      # Диффузия текста (Diffusion-LM, Genie...)
│   └── custom/                    # Шаблон: пользователь копирует → 40 строк → готово
│
├── assemblers/                    # ★ СБОРЩИКИ (готовые конструкции)
│   ├── model_assembler.py         # ModelAssembler.from_config() → ModularDiffusionModel
│   ├── pipeline_assembler.py      # PipelineAssembler (generation, training, distillation)
│   ├── adapter_assembler.py       # Автоматически приклеивает адаптеры
│   └── multi_modal_assembler.py   # Собирает цепочки модальностей (text → image → video)
│
├── training/                      # Обучение — тоже Lego
│   ├── trainer.py                 # ModularTrainer (один на всё)
│   ├── strategies/                # FullFinetune, LoRAOnly, AdapterOnly, Curriculum...
│   ├── losses/                    # Универсальные + модальные
│   └── datasets/                  # AbstractDataset + HF + WebDataset + Synthetic
│
├── deployment/                    # Развёртывание
│   ├── server/                    # FastAPI + streaming + queue
│   ├── docker/                    # multi-stage + flash-attn + xformers
│   ├── cloud/                     # RunPod, Vast, Modal, Lambda adapters
│   └── export/                    # ONNX, TensorRT, GGUF, OpenVINO
│
├── ui/                            # Интерфейсы
│   ├── gradio/                    # Динамический: подстраивается под любой plugin
│   ├── components/                # BlockSelector, SlotConnector, LivePreview...
│   └── app.py                     # gradio_app.launch(model) → готовый UI
│
├── integration/                   # Импорт из внешнего мира
│   ├── diffusers.py               # from_pretrained("black-forest-labs/FLUX.1-dev")
│   ├── comfyui.py                 # Импорт workflow'ов
│   └── peft.py                    # Конверсия LoRA
│
├── configs/                       # Конфиги = инструкции Lego
│   ├── blocks/                    # базовые кирпичики
│   ├── plugins/                   # image_flux.yaml, timeseries_forecast.yaml
│   ├── recipes/                   # Полные наборы (molecule_design_with_control.yaml)
│   └── user/                      # .gitignored — твои сборки
│
├── examples/                      # От "один кирпичик" до "монстра"
│   ├── lego_01_hello.py           # Самая простая сборка
│   ├── lego_02_custom_block.py    # Новый solver за 20 строк
│   ├── lego_03_new_plugin.py      # Новая модальность
│   ├── lego_04_frankenstein.py    # Текст → 3D → аудио
│   └── lego_05_full_pipeline.py
│
├── tools/
│   ├── block_inspector.py         # Показывает все доступные блоки
│   ├── graph_visualizer.py        # Рисует граф модели (Mermaid + Graphviz)
│   └── benchmark.py               # Сравнение сборок
│
├── tests/
├── docs/
├── scripts/
├── requirements/
├── pyproject.toml
└── README.md                      # "Собери свою диффузию как Lego. Даже если её ещё не придумали."