# Поддержка языковых моделей (LLM)

**Назначение:** полное и детальное описание того, **как фреймворк Иггдрасиль обеспечивает работу с языковыми моделями (LLM)** в рамках единого движка гиперграфа. Документ задаёт **полный паритет по возможностям** с эталонными решениями: **Hugging Face Transformers v5.3.0** (пайплайны, генерация, токенизация, квантизация), **LangChain** (create_agent, middleware), **LangGraph** (StateGraph, Pregel, каналы, чекпоинт), **DeepAgents** (глубокий агент с планированием, файловой системой, субагентами). Паритет достигается **полной интеграцией аналогичного функционала на базе движка Иггдрасиль** — без использования кода или исполнения этих библиотек как основы; собственные блоки, гиперграфы и run принадлежат фреймворку. Документ описывает маппинг каждого компонента на роли узлов-задач, типичные топологии графов, форматы данных на портах, цикл генерации и агентный цикл, конфигурацию, обучение, сериализацию и использование на уровне воркфлоу.

**Связь с каноном:** [01_FOUNDATION.md](01_FOUNDATION.md) — Block, Node, порты, run; [02_ABSTRACT_TASK_NODES.md](02_ABSTRACT_TASK_NODES.md) — семь ролей и контракты портов; [03_TASK_HYPERGRAPH.md](03_TASK_HYPERGRAPH.md) — гиперграф задачи, циклы, agent_loop; [HYPERGRAPH_ENGINE.md](HYPERGRAPH_ENGINE.md) — планировщик, итеративная фаза, буферы; [DOMAINS_DEPLOYMENT_TRAINING.md](DOMAINS_DEPLOYMENT_TRAINING.md) — обзор доменов.

**Язык:** русский.

**Паритет и превосходство по удобству.** Решения на базе Иггдрасиль должны не просто обеспечивать **паритет по функциональности и мощности** со всеми перечисленными state-of-the-art решениями (Transformers, LangChain, LangGraph, DeepAgents), **но и** давать пользователю **более удобный, простой и интуитивно понятный интерфейс взаимодействия**: любое решение (completion, chat, RAG, агент с инструментами, структурированный вывод) должно быть собираемо **в 3–5 строк кода** при сохранении возможности глубочайшей кастомизации. Это достигается продуманной логикой устройства фреймворка и разумной автоматизацией (фабрики гиперграфов, разумные умолчания, единый контракт run), без ущерба для расширяемости и полного контроля над каждым узлом и параметром.

---

## 1. Место языковых моделей в фреймворке

### 1.1 Единая онтология

Языковые модели **не вводят отдельной онтологии**. Они реализуются **теми же сущностями**, что и остальные домены: узлы-задачи (Backbone, Conjector, Inner Module, Outer Module, Converter, Injector, Helper), гиперграф задачи, **движок гиперграфа** (планировщик, итеративная фаза, буферы), порты и run. Один гиперграф задачи = **одна задача по LLM** (completion, chat, RAG, агент с инструментами, структурированный вывод); комбинирование задач (RAG → генерация → постобработка, LLM → диффузия) выполняется на уровне **воркфлоу**, где узлами служат целые гиперграфы. Движок не различает «тип задачи» — только структуру графа, порты и число итераций цикла; цикл генерации токенов и агентный цикл (модель ↔ инструменты) выражаются одной и той же механикой итеративной фазы ([HYPERGRAPH_ENGINE.md](HYPERGRAPH_ENGINE.md)).

### 1.2 Что такое «задача по языковой модели» в каноне

**Задача по языковой модели** — это задача генерации или обработки текста, в которой центральным элементом является **модель предсказания следующего токена** (трансформер decoder-only или encoder-decoder). Вход — промпт (и опционально системный промпт, история сообщений, документы RAG); выход — сгенерированный текст и опционально структурированные вызовы инструментов (tool_calls). Генерация **итеративна**: на каждом шаге цикла Backbone выдаёт логиты; Inner Module выполняет сэмплирование (greedy, temperature, top_p, top_k и т.д.), обновление состояния (KV cache, позиция) и проверку остановки (EOS, stop-токены, max_length); цикл повторяется до num_loop_steps (max_new_tokens) или до сигнала остановки от Inner Module.

В рамках фреймворка эта задача **полностью укладывается** в гиперграф: **начальная фаза** (Converter — токенизатор; Conjector — форматирование промпта/чата; Outer Module — prefill и начальное состояние для цикла) → **циклическая фаза** (Backbone ↔ Inner Module до K итераций) → **конечная фаза** (Converter — детокенизация → внешний выход `text`). Движок строит итеративный план и выполняет цикл без знания о том, что это «LLM»; контракт портов и буферов единообразен. Когда LLM выступает ядром **агента**, тот же граф встраивается в узел-агент, а инструменты — в отдельные узлы; движок реализует **agent_loop**: вызов агента → при наличии tool_calls выполнение узлов-инструментов → сбор tool_results → повторный вызов агента ([03_TASK_HYPERGRAPH.md](03_TASK_HYPERGRAPH.md) §7.3).

### 1.3 Связь с эталонными решениями

- **Transformers** даёт эталон **пайплайна генерации**: Pipeline (preprocess → model.generate → postprocess), GenerationConfig, логит-процессоры, критерии остановки, стримеры, KV cache, continuous batching, assisted/speculative decoding. В Иггдрасиль этому соответствует один гиперграф с ролями Converter, Conjector, Outer Module, Backbone, Inner Module, Converter (детокенизация); «pipeline» = run(hypergraph, inputs).
- **LangGraph** даёт эталон **графа состояний**: узлы пишут в общее состояние (State → Partial<State>); планировщик определяет следующие узлы; каналы с редьюсерами, чекпоинт, прерывания, resume. В Иггдрасиль гиперграф задачи с циклом и буферами движка реализует ту же семантику: узлы обмениваются данными по портам, движок ведёт буферы и итеративную фазу; чекпоинт и персистенция состояния задаются сериализацией графа и опциями run ([SERIALIZATION.md](SERIALIZATION.md)).
- **LangChain create_agent** и **DeepAgents create_deep_agent** — фабрики готового агентного графа (модель, инструменты, system_prompt, middleware, checkpointer, store, interrupt_before/after, subagents, backend). В Иггдрасиль аналог — **сборка гиперграфа по конфигу**: узел-агент (Backbone = LLM + логика tool_calls), узлы-инструменты, опционально Helper (RAG, файлы), Conjector (формат, память); конфиг задаёт состав узлов, рёбра, metadata (в т.ч. interrupt_on, checkpointer_ref); «middleware» выражается дополнительными узлами или опциями run.

---

## 2. Полный паритет с эталонными решениями

Ниже перечислено **всё**, что фреймворк Иггдрасиль должен уметь поддерживать в части языковых моделей и агентов, чтобы обеспечить **полный паритет** по возможностям с:

- **Hugging Face Transformers v5.3.0** — пайплайны текстовой генерации, конфигурация генерации, процессоры логитов, критерии остановки, стриминг, кэш, батчинг, квантизация, обучение.
- **LangChain** — create_agent (модель, инструменты, system_prompt, middleware, response_format, checkpointer, store, interrupt_before/after).
- **LangGraph** — StateGraph, Pregel, каналы состояния, чекпоинт, store, прерывания, resume, стриминг.
- **DeepAgents** — глубокий агент (todo, файловая система, субагенты, skills, memory, backend, interrupt_on, MCP).

**Важно:** паритет достигается **не за счёт использования кода или исполнения этих библиотек**, а за счёт **полной интеграции аналогичного функционала в рамках фреймворка Иггдрасиль**. Блоки (Backbone для LLM, Inner Module для шага сэмплирования/остановки, Converter для токенизации, Conjector для формата промпта, узлы-инструменты и т.д.), движок гиперграфа (планировщик, итеративная фаза, agent_loop), сериализация (конфиг + чекпоинт) и API run принадлежат Иггдрасиль. Допускается совместимость по форматам (загрузка весов из Hugging Face Hub, совместимость протоколов API); исполнение и архитектура остаются в границах фреймворка.

**Источник перечисления:** репозитории Hugging Face Transformers, LangChain, LangGraph, DeepAgents (актуальные на момент подготовки документа: generation, pipelines, models/auto; agents/factory, middleware; pregel, graph, channels, checkpoint; graph.py, server_graph, middleware).

---

### 2.1 Паритет с Transformers v5.3.0: генерация и пайплайны

Фреймворк должен позволять выражать через гиперграф и блоки **всё** нижеперечисленное.

**Конфигурация генерации (GenerationConfig):**

- Параметры: `max_new_tokens`, `max_length`, `min_length`, `min_new_tokens`, `do_sample`, `temperature`, `top_p`, `top_k`, `typical_p`, `epsilon_cutoff`, `eta_cutoff`, `repetition_penalty`, `encoder_repetition_penalty`, `length_penalty`, `no_repeat_ngram_size`, `bad_words_ids`, `force_words_ids`, `num_beams`, `num_beam_groups`, `diversity_penalty`, `num_return_sequences`, `output_scores`, `output_logits`, `output_attentions`, `output_hidden_states`, `return_dict_in_generate`, `use_cache`, `eos_token_id`, `pad_token_id`, `prefix_allowed_tokens_fn`, `suppress_tokens`, `begin_suppress_tokens`, `assistant_model`, `prompt_lookup_num_tokens`, и др. — задаются конфигом Inner Module (sampler) и опциями run; при необходимости часть параметров передаётся в Backbone (например, use_cache).

**Режимы генерации (GenerationMode):**

- **Sample** (do_sample=True) — сэмплирование по температуре/top_p/top_k; реализуется в Inner Module (sampler).
- **Greedy** (do_sample=False) — argmax; тот же Inner Module.
- **Beam search** — многолучевой поиск; Inner Module может быть расширен или выделен отдельный тип (inner_module/beam_search) с накоплением лучей и выбором лучшего.
- **Assisted / speculative decoding** — кандидаты от вспомогательной модели или lookup; выражается опциональным узлом (CandidateGenerator-подобный) или конфигом Inner Module с ссылкой на assistant_backbone.

**Логит-процессоры и варперы (LogitsProcessor / LogitsWarper):**

- В Transformers: `LogitsProcessor`, `LogitsProcessorList`, `TemperatureLogitsWarper`, `TopKLogitsWarper`, `TopPLogitsWarper`, `TopHLogitsWarper`, `TypicalLogitsWarper`, `EpsilonLogitsWarper`, `EtaLogitsWarper`, `MinPLogitsWarper`, `RepetitionPenaltyLogitsProcessor`, `NoRepeatNGramLogitsProcessor`, `EncoderRepetitionPenaltyLogitsProcessor`, `EncoderNoRepeatNGramLogitsProcessor`, `MinLengthLogitsProcessor`, `MinNewTokensLengthLogitsProcessor`, `ForcedBOSTokenLogitsProcessor`, `ForcedEOSTokenLogitsProcessor`, `NoBadWordsLogitsProcessor`, `PrefixConstrainedLogitsProcessor`, `SequenceBiasLogitsProcessor`, `SuppressTokensLogitsProcessor`, `SuppressTokensAtBeginLogitsProcessor`, `ClassifierFreeGuidanceLogitsProcessor`, `WatermarkLogitsProcessor`, `InfNanRemoveLogitsProcessor`, `LogitNormalization`, и др. В Иггдрасиль все они — **часть конфига Inner Module (sampler)** или цепочка процессоров, применяемая в одном узле Inner Module к logits перед сэмплированием; при необходимости отдельный подтип inner_module/logits_processor.

**Критерии остановки (StoppingCriteria):**

- `MaxLengthCriteria`, `MaxTimeCriteria`, `EosTokenCriteria`, `StopStringCriteria`, `ConfidenceCriteria`, `StoppingCriteriaList`. В Иггдрасиль — логика внутри Inner Module на каждом шаге цикла: после выбора next_token_id проверяется условие остановки; при срабатывании Inner Module возвращает флаг «завершить цикл», движок выходит из циклической фазы.

**Стримеры (Streamers):**

- `BaseStreamer`, `TextIteratorStreamer`, `TextStreamer`, `AsyncTextIteratorStreamer`. В Иггдрасиль — опция run(..., stream=True) или callback в options; движок или блок при каждой итерации цикла (или при каждом новом токене) вызывает callback с частичным результатом; контракт внешнего выхода (порт `text`) не меняется, способ доставки — итератор или callback.

**Кандидаты и assisted decoding (CandidateGenerator):**

- `AssistedCandidateGenerator`, `EarlyExitCandidateGenerator`, `PromptLookupCandidateGenerator`, `UniversalSpeculativeDecodingGenerator`. Выражаются конфигом Inner Module или отдельным узлом, выдающим кандидаты для следующего шага; Backbone при необходимости принимает assistant_model по порту или по конфигу.

**Continuous batching:**

- `ContinuousBatchingManager`, `ContinuousMixin`, `Scheduler` (FIFO, PrefillFirst). В Иггдрасиль — опция run или конфиг движка/Outer Module: батч запросов обрабатывается с общим планировщиком шагов; реализация может быть в слое выполнения (не меняя контракта гиперграфа).

**Кэш (KV cache):**

- Имена кэша в Transformers: `past_key_values`, `cache_params`, `state`, `mems`, `past_buckets_states`. В Иггдрасиль Backbone имеет опциональные вход/выход `past_key_values`; Outer Module при prefill заполняет кэш; на каждой итерации цикла Inner Module передаёт обновлённый кэш обратно в Backbone. Типы кэша (DynamicCache, StaticCache, EncoderDecoderCache, QuantizedCache) задаются конфигом Backbone.

**Watermarking:**

- `WatermarkingConfig`, `WatermarkLogitsProcessor`, `WatermarkDetector`, `SynthIDTextWatermarkDetector`. Реализуются в Inner Module (модификация логитов) и опционально в Helper (детектор); конфиг по соглашению.

**Пайплайн текстовой генерации (TextGenerationPipeline):**

- В Transformers: единая точка входа pipeline("text-generation", model=...); вход — строка или список сообщений чата; выход — generated_text; параметры return_full_text, return_tensors, clean_up_tokenization_spaces, prefix, handle_long_generation, stop_sequence, truncation, max_length, continue_final_message, tools, documents. В Иггдрасиль — один гиперграф с exposed_inputs (prompt или messages), exposed_outputs (text); опции run соответствуют параметрам пайплайна; decoder-only левый padding для батчей задаётся конфигом Converter (tokenizer).

**Токенизация:**

- `PreTrainedTokenizerBase`, `BatchEncoding`, `AddedToken`, `PreTrainedTokenizerFast`, SentencePiece backend, Python backend; конвертация текст ↔ token_ids, attention_mask, position_ids. В Иггдрасиль — блоки Converter (tokenizer, detokenizer); конфиг задаёт путь к словарю/модели токенизатора, max_length, padding_side.

**Модели (Auto):**

- Transformers предоставляет `AutoModelForCausalLM`, `AutoModelForSeq2SeqLM` и маппинги по архитектурам (Llama, GPT-2, Mistral, Qwen, T5, и сотни других). В Иггдрасиль Backbone (LLM) загружается по конфигу (checkpoint_ref, или provider/model_id для API); архитектура задаётся block_type или config; один и тот же гиперграф может работать с разными моделями при замене конфига Backbone.

**Квантизация:**

- Конфиги: BitsAndBytes, GPTQ, AWQ, Aqlm, Quanto, и др. В Иггдрасиль — конфиг блока Backbone (quantization_config или аналог); загрузчик весов применяет квантизацию при инициализации; исполнение остаётся в рамках фреймворка.

**Trainer и обучение:**

- TrainingArguments, цикл обучения, callbacks, оптимизатор, чекпоинты. В Иггдрасиль обучение на уровне графа/блока по [DOMAINS_DEPLOYMENT_TRAINING.md](DOMAINS_DEPLOYMENT_TRAINING.md) §6; trainable узлы, loss по выходам, backward, сохранение state_dict.

Итог по §2.1: каждый перечисленный элемент Transformers (генерация, пайплайн, токенизация, кэш, квантизация) должен быть **выразим** конфигом и узлами гиперграфа Иггдрасиль и выполняться движком фреймворка.

---

### 2.2 Паритет с LangChain: create_agent и middleware

**create_agent (фабрика агента):**

- Параметры: `model` (строка или BaseChatModel), `tools` (список BaseTool | Callable | dict), `system_prompt`, `middleware` (последовательность AgentMiddleware), `response_format`, `state_schema`, `context_schema`, `checkpointer`, `store`, `interrupt_before`, `interrupt_after`, `debug`, `name`, `cache`. Возвращает `CompiledStateGraph`. В Иггдрасиль аналог — **фабрика или конфиг гиперграфа**: узел-агент (Backbone = LLM + интерпретация tool_calls), узлы-инструменты (каждый — узел-задача с портами по аргументам инструмента), Conjector для system_prompt и формата сообщений; state_schema/context_schema — схема портов и metadata; checkpointer/store — сериализация и опции run ([SERIALIZATION.md](SERIALIZATION.md)); interrupt_before/after — metadata гиперграфа или опции run (движок при достижении указанного узла может приостановить и вернуть управление или вызвать callback).

**Middleware:**

- Типы: TodoListMiddleware, SummarizationMiddleware, HumanInTheLoopMiddleware, ToolRetryMiddleware, ToolCallLimitMiddleware, ShellToolMiddleware, FilesystemMiddleware, SubAgentMiddleware, PatchToolCallsMiddleware, SkillsMiddleware, AnthropicPromptCachingMiddleware, и др. В Иггдрасиль middleware **не отдельная сущность**, а выражается: (1) дополнительными узлами в графе (например, Helper для summarization, узел для todo list); (2) обёрткой вокруг вызова модели (инъекция в контекст, постобработка tool_calls) — может быть частью узла-агента или отдельным Conjector/Inner Module; (3) опциями run (interrupt_on, max_tool_calls, retry_policy). Последовательность middleware задаётся порядком узлов и конфигом.

**ResponseFormat (структурированный вывод):**

- В LangChain — конфигурация формата ответа (JSON, Pydantic). В Иггдрасиль — Conjector (инструкция по формату в промпт) и/или постобработка в Converter/Helper (парсинг JSON из текста или из специальных токенов).

Итог по §2.2: сценарий «create_agent с моделью, инструментами, system_prompt, middleware, checkpointer, interrupt» должен быть собираем как один гиперграф Иггдрасиль с соответствующими узлами и metadata; выполнение — run(hypergraph, inputs) с опциями.

---

### 2.3 Паритет с LangGraph: StateGraph, Pregel, каналы, чекпоинт

**StateGraph:**

- Граф, узлы которого имеют сигнатуру State → Partial<State>; опционально редьюсеры по ключам состояния. В Иггдрасиль гиперграф задачи: каждый узел имеет входные и выходные порты; «состояние» — совокупность буферов движка, соответствующих портам; обновление состояния = запись выхода узла в буфер и передача по рёбрам следующим узлам. Редьюсер (агрегация записей от нескольких узлов в один слот) при необходимости реализуется узлом-агрегатором или конвенцией движка (например, append для списка сообщений).

**Pregel (цикл выполнения):**

- Планирование следующих задач по текущему состоянию и графу → выполнение узлов → применение записей (writes) в каналы → чекпоинт/стрим; поддержка прерываний и resume. В Иггдрасиль это **итеративная фаза движка** ([HYPERGRAPH_ENGINE.md](HYPERGRAPH_ENGINE.md)): планировщик определяет следующий шаг цикла (Backbone + Inner Module в случае генерации; или узел-агент + узлы-инструменты в agent_loop); выполнение узлов, обновление буферов, проверка условия выхода из цикла; при включённой персистенции — сохранение чекпоинта по соглашению.

**Каналы (Channels):**

- LastValue, EphemeralValue, AnyValue, Topic, NamedBarrierValue, редьюсеры (binop). В Иггдрасиль «канал» — именованный буфер/порт; семантика «последнее значение» или «редьюс» задаётся конфигом графа или типом порта; эфемерные значения не записываются в чекпоинт — опция в metadata порта или узла.

**Checkpoint и Store:**

- BaseCheckpointSaver, Checkpoint, сериализация (msgpack, jsonplus, encrypted); in-memory, PostgreSQL; BaseStore для персистентного хранилища. В Иггдрасиль чекпоинт = state_dict графа + опционально пользовательское состояние (state) по [SERIALIZATION.md](SERIALIZATION.md); store — внешний сервис или Helper, с которым граф взаимодействует по портам; конфиг задаёт, куда писать чекпоинт и как восстанавливать сессию (thread_id, checkpoint_id — в опциях run).

**Прерывания (Interrupt) и resume:**

- Остановка до/после указанного узла; возобновление с того же состояния. В Иггдрасиль interrupt_on в metadata или run options; движок при достижении узла (или после выполнения инструмента в agent_loop) может вернуть частичный результат и флаг «приостановлено»; следующий run с теми же thread_id/checkpoint_id продолжает с сохранённого состояния.

**Managed values (например, is_last_step):**

- Значения, управляемые рантаймом. В Иггдрасиль — опционально передаются в run config (например, step_index, is_final_step) и доступны узлам по конвенции (порт или config в run).

Итог по §2.3: семантика StateGraph + Pregel (состояние, цикл, каналы, чекпоинт, прерывания) должна быть достижима **движком гиперграфа Иггдрасиль** и конфигом графа без использования кода LangGraph.

---

### 2.4 Паритет с DeepAgents: глубокий агент, инструменты, субагенты

**create_deep_agent:**

- Параметры: model, tools, system_prompt, middleware, subagents, skills, memory, response_format, context_schema, checkpointer, store, backend, interrupt_on, debug, name, cache. По умолчанию — модель Claude, StateBackend; встроенные инструменты: write_todos, ls, read_file, write_file, edit_file, glob, grep, execute, task (субагенты). В Иггдрасиль — гиперграф с узлом-агентом (Backbone = LLM), узлами-инструментами (каждый инструмент = узел с портами по аргументам); backend (файловая система, sandbox) реализуется как Helper или отдельные узлы, имеющие доступ к хранилищу по конфигу; write_todos, ls, read_file и т.д. — типы узлов-инструментов (block_type, например, helper/filesystem_read, helper/filesystem_write, helper/todo, helper/execute).

**Subagents:**

- Список SubAgent (name, description, system_prompt, tools, model, middleware). В Иггдрасиль субагент — **вложенный гиперграф** или отдельный узел-агент с собственным конфигом; инструмент «task» в графе главного агента вызывает этот подграф (движок выполняет run подграфа с соответствующими входами); результат возвращается как tool_result главному агенту.

**Skills и memory:**

- Skills — источники навыков (файлы, пути); memory — файлы памяти (например, AGENTS.md), подставляемые в system prompt. В Иггдрасиль Helper загружает навыки/память по конфигу; Conjector собирает системный промпт с подстановкой содержимого из Helper; либо отдельный узел формирует полный промпт с памятью.

**Backend (StateBackend, Sandbox):**

- StateBackend — состояние в памяти; Sandbox — файловая система и выполнение команд. В Иггдрасиль узлы-инструменты (read_file, write_file, execute) получают backend через конфиг (например, ссылка на хранилище или sandbox-провайдер); при run данные передаются по портам, backend вызывается внутри блока.

**interrupt_on (tool → config):**

- Приостановка перед выполнением указанного инструмента (например, edit_file) для подтверждения пользователем. В Иггдрасиль — metadata (interrupt_on: [edit_file]) или опции run; движок при планировании вызова узла edit_file может вызвать callback и приостановить выполнение до следующего run с переданным решением.

**MCP (Model Context Protocol):**

- Инструменты, загружаемые из внешней конфигурации MCP. В Иггдрасиль MCP-инструменты — узлы-инструменты, конфиг которых указывает источник (MCP server); при инициализации графа или при run инструменты резолвятся и вызываются по tool_id; контракт портов единый.

**Server (langgraph dev) и конфиг из env:**

- DeepAgents CLI формирует граф для сервера из переменных окружения (ServerConfig.from_env()). В Иггдрасиль развёртывание графа на сервере и загрузка конфига из env — задача развёртывания ([DOMAINS_DEPLOYMENT_TRAINING.md](DOMAINS_DEPLOYMENT_TRAINING.md) §6); конфиг гиперграфа может быть сериализован и подставлен из env при инициализации.

Итог по §2.4: сценарий «глубокий агент с todo, файлами, субагентами, skills, memory, backend, interrupt_on, MCP» должен быть собираем как гиперграф Иггдрасиль с соответствующими узлами и конфигом; исполнение — движок с поддержкой agent_loop и опций run.

---

### 2.5 Сводная таблица паритета (чеклист)

| Источник | Возможность | Выражение в Иггдрасиль |
|----------|-------------|------------------------|
| **Transformers** | GenerationConfig (max_new_tokens, temperature, top_p, do_sample, и т.д.) | Конфиг Inner Module (sampler), опции run |
| **Transformers** | LogitsProcessor / LogitsWarper (temperature, repetition, no_bad_words, и т.д.) | Цепочка в Inner Module или конфиг sampler |
| **Transformers** | StoppingCriteria (EOS, max_length, stop_string) | Логика в Inner Module, флаг выхода из цикла |
| **Transformers** | Streamers (TextIteratorStreamer и др.) | run(..., stream=True) или callback в options |
| **Transformers** | KV cache (past_key_values, DynamicCache, StaticCache) | Порты Backbone, Outer Module prefill, передача по циклу |
| **Transformers** | Continuous batching, Scheduler | Опции run или конфиг движка/Outer Module |
| **Transformers** | Assisted / speculative decoding | Конфиг Inner Module или узел-кандидат |
| **Transformers** | TextGenerationPipeline, chat (messages) | Гиперграф с exposed_inputs (prompt/messages), exposed_outputs (text) |
| **Transformers** | Токенизаторы (PreTrainedTokenizerBase, Fast, SentencePiece) | Converter (tokenizer, detokenizer) |
| **Transformers** | AutoModelForCausalLM, маппинг архитектур | Backbone (LLM) по конфигу checkpoint_ref / block_type |
| **Transformers** | Квантизация (BitsAndBytes, GPTQ, и др.) | Конфиг Backbone quantization_config |
| **Transformers** | Trainer, обучение | trainable узлы, loss, backward, state_dict ([DOMAINS_DEPLOYMENT_TRAINING.md](DOMAINS_DEPLOYMENT_TRAINING.md) §6) |
| **LangChain** | create_agent (model, tools, system_prompt, middleware, …) | Фабрика/конфиг гиперграфа: узел-агент + узлы-инструменты + Conjector + metadata |
| **LangChain** | AgentMiddleware (todo, summarization, HITL, tool_retry, filesystem, subagents, и т.д.) | Доп. узлы, обёртки в узле-агенте, опции run |
| **LangChain** | response_format, state_schema, context_schema | Conjector (формат), схема портов и metadata |
| **LangChain** | checkpointer, store | Сериализация графа, опции run (thread_id, checkpoint_id), Helper/Store по конфигу |
| **LangChain** | interrupt_before, interrupt_after | metadata или run options, движок приостанавливает и возвращает |
| **LangGraph** | StateGraph (State → Partial<State>) | Гиперграф, порты = ключи состояния, буферы движка |
| **LangGraph** | Pregel (цикл: планирование → выполнение → writes → checkpoint) | Итеративная фаза движка, буферы, опционально чекпоинт |
| **LangGraph** | Каналы (LastValue, reducer, barrier, ephemeral) | Типы портов/буферов по конфигу или конвенции |
| **LangGraph** | Checkpoint, Store, serde | state_dict графа, опции сохранения/загрузки ([SERIALIZATION.md](SERIALIZATION.md)) |
| **LangGraph** | Interrupt, resume | interrupt_on в metadata/run, run с checkpoint_id/thread_id для продолжения |
| **DeepAgents** | create_deep_agent (backend, subagents, skills, memory, interrupt_on) | Гиперграф с backend в конфиге узлов, субагенты = вложенные графы, Helper для skills/memory |
| **DeepAgents** | Встроенные инструменты (write_todos, ls, read_file, write_file, edit_file, glob, grep, execute, task) | Узлы-инструменты с block_type helper/todo, helper/filesystem_*, helper/execute; task → run подграфа |
| **DeepAgents** | Middleware (TodoList, Filesystem, Summarization, PatchToolCalls, Skills, HITL) | Узлы и конфиг узла-агента (инъекция контекста, постобработка tool_calls) |
| **DeepAgents** | MCP tools | Узлы-инструменты с конфигом источника MCP |
| **DeepAgents** | Server config from env | Развёртывание: загрузка конфига из env при инициализации графа |

Итог: **полный паритет** означает, что для каждой перечисленной возможности эталонных решений в таблице указано, как она отображается на гиперграф, блоки, порты, metadata или опции run в Иггдрасиль; реализация обеспечивается **движком и блоками фреймворка**, а не вызовом этих библиотек.

---

## 3. Маппинг компонентов LLM и агента на роли узлов-задач

### 3.1 Сводная таблица

| Компонент | Роль в каноне | block_type (пример) | Назначение |
|-----------|----------------|----------------------|------------|
| Модель языка (предсказание следующего токена) | **Backbone** | `backbone/llm`, `backbone/transformer_lm` | (input_ids, attention_mask, position_ids, past_key_values?) → logits, past_key_values? |
| Шаг сэмплирования, процессоры логитов, остановка | **Inner Module** | `inner_module/sampler`, `inner_module/step` | (logits, state) → next_token_id, updated_state; флаг stop; применение LogitsProcessor/StoppingCriteria |
| Системный промпт, формат чата, подстановка RAG/памяти | **Conjector** | `conjector/prompt_format`, `conjector/context` | (messages, system_prompt, documents?) → отформатированный промпт или condition |
| Токенизатор / детокенизатор | **Converter** | `converter/tokenizer`, `converter/detokenizer` | текст ↔ token_ids, attention_mask |
| Prefill, начальное состояние цикла | **Outer Module** | `outer_module/prefill`, `outer_module/context` | token_ids → initial state (input_ids для первого шага или cache после prefill) |
| LoRA, адаптеры для LLM | **Injector** | `injector/lora` | Встраивается в Backbone; веса влияют на forward |
| RAG, API, файлы, todo, execute | **Helper** | `helper/rag`, `helper/api`, `helper/filesystem`, `helper/todo`, `helper/execute` | Поиск документов, вызов API, чтение/запись файлов, список задач, выполнение команд |
| Узел-агент (LLM + tool_calls) | Специализация графа | — | Внутри: Backbone (LLM) + логика парсинга tool_calls; снаружи: порты prompt/context, выход response + tool_calls; движок выполняет agent_loop |
| Инструмент (tool) | Узел-задача | Зависит от инструмента | Входы по аргументам инструмента; выход — результат; вызывается движком по tool_id при agent_loop |

### 3.2 Backbone (ядро LLM)

- **Входы:** `input_ids`, `attention_mask`, опционально `position_ids`, `past_key_values`, `condition` (если контекст от Conjector отдельным портом).
- **Выходы:** `logits`; опционально `past_key_values` для следующей итерации.
- **Семантика:** один вызов = один forward; сэмплирование и остановка — в Inner Module. Поддержка decoder-only и encoder-decoder по конфигу (архитектура, block_type).
- **Режим API:** при backend: "api", provider, model_id блок при run выполняет запрос к внешнему API и возвращает logits/токены по контракту портов; контракт для графа тот же.

### 3.3 Inner Module (шаг генерации)

- **Входы:** `logits`, текущее состояние (накопленные token_ids, position, past_key_values при необходимости).
- **Выходы:** `next_token_id`, обновлённое состояние; флаг или сигнал движку «завершить цикл».
- **Логика:** применение цепочки LogitsProcessor/LogitsWarper (temperature, top_p, top_k, repetition_penalty, no_repeat_ngram, и т.д.); сэмплирование; проверка EOS, stop_token_ids, max_length, stop_string; обновление KV cache и позиции. Конфиг задаёт полный набор процессоров и критериев остановки.

### 3.4 Conjector (контекст и формат промпта)

- **Вход:** сырой промпт, messages, system_prompt, документы от Helper (RAG), память (skills, memory).
- **Выход:** одна строка промпта или структура для Converter/Outer Module; либо condition для Backbone при отдельном канале контекста.
- **Выполнение:** один раз до цикла (или один раз на run). Шаблоны чата (ChatML, Alpaca, и т.д.) задаются конфигом.

### 3.5 Outer Module (начальное состояние цикла)

- **Вход:** token_ids (и опционально attention_mask) от Converter.
- **Выход:** initial input_ids (или после prefill — past_key_values + position для продолжения). При prefill один проход Backbone на весь промпт; результат кэша передаётся в цикл.
- **Связь:** выход подаётся на первый шаг цикла (Backbone); далее цикл Backbone ↔ Inner Module.

### 3.6 Converter (токенизация и детокенизация)

- Токенизатор: текст → token_ids, attention_mask; padding_side, max_length по конфигу (для decoder-only батчей обычно left).
- Детокенизатор: итоговая последовательность token_ids → текст; clean_up_tokenization_spaces по конфигу.

### 3.7 Injector (LoRA и адаптеры)

- Веса встраиваются в Backbone; при run Backbone использует базовые веса + LoRA. Condition (adapter_id при multi-LoRA) опционально по порту. Обучаемые параметры — только LoRA; state_dict сохраняется отдельно или вместе с Backbone.

### 3.8 Helper (RAG, API, файлы, todo, execute)

- RAG: запрос → список документов; выход в Conjector. API: вызов внешнего сервиса по портам. Filesystem: read_file, write_file, edit_file, ls, glob, grep по конфигу backend. Todo: write_todos, чтение списка задач. Execute: выполнение команд (sandbox) по конфигу. Каждый — узел с соответствующими портами.

### 3.9 Агент и инструменты

- **Узел-агент:** внутри — подграф или логика «LLM + парсинг tool_calls»; входы — prompt/context, опционально tool_results; выходы — response, tool_calls (список {tool_id, arguments}). Движок по конфигу (tool_id → node_id) при наличии tool_calls вызывает узлы-инструменты, собирает tool_results и снова вызывает агента (agent_loop) до отсутствия tool_calls или max_steps.
- **Узел-инструмент:** порты = аргументы инструмента; один выход — результат; block_type или node_id сопоставляется с tool_id в конфиге агента.

---

## 4. Типичные графы (топологии)

### 4.1 Text completion (дозаполнение)

Поток: внешний вход `prompt` → Converter (tokenizer) → Outer Module (initial state / prefill) → **цикл** (Backbone → logits → Inner Module → next_token_id, state; повтор до K или stop) → Converter (detokenizer) → внешний выход `text`. Узлы: tokenizer, prefill, llm, sampler, detokenizer; опционально Conjector (формат), Injector (LoRA), Helper (RAG).

### 4.2 Chat (диалог)

Отличие: внешние входы `messages` (и опционально `system_prompt`). Conjector формирует строку в формате чата; далее как completion. Выход — ответ ассистента (text).

### 4.3 RAG

Внешний вход `query` (и опционально history). Helper (RAG) по query возвращает документы; Conjector вставляет их в шаблон (например, «Context: … Question: …»); далее tokenizer → prefill → цикл → detokenizer → `text`.

### 4.4 Агент с LLM и инструментами

Гиперграф содержит узел-агент (Backbone = LLM + парсинг tool_calls) и узлы-инструменты. Внешний вход — prompt/context. Движок выполняет agent_loop: run(агент) → при tool_calls run(инструменты по tool_id) → tool_results → снова run(агент); выход — финальный response. Соответствует create_agent (LangChain) и create_deep_agent (DeepAgents) при соответствующем наборе узлов и конфиге.

### 4.5 Глубокий агент (DeepAgents-подобный)

Тот же граф, что §4.4, но с узлами-инструментами: write_todos, ls, read_file, write_file, edit_file, glob, grep, execute, task (вызов субагента). Backend (файлы, sandbox) задаётся конфигом этих узлов. Conjector собирает system_prompt с подстановкой memory/skills из Helper. interrupt_on задаётся metadata или run options. Субагент для task — вложенный гиперграф; при вызове task движок выполняет run(субагент, inputs) и возвращает результат как tool_result.

### 4.6 Один шаг (без цикла)

Эмбеддинг или один forward: Backbone вызывается один раз, Inner Module не используется; num_loop_steps = 0 или цикл отсутствует. Выход — logits или эмбеддинги.

---

## 5. Форматы данных на портах

### 5.1 input_ids, attention_mask, position_ids

- Тензоры формы [B, seq_len]; типы по соглашению реализации (int32/int64). position_ids — для RoPE и аналогов.

### 5.2 logits

- [B, seq_len, vocab_size] или [B, vocab_size] для последней позиции. Inner Module применяет процессоры и сэмплирование.

### 5.3 past_key_values (KV cache)

- Структура по конфигу Backbone (DynamicCache, StaticCache, список пар key/value по слоям). Передаётся между итерациями цикла.

### 5.4 text (внешние границы)

- Строка или список строк (batch). Конвертация в token_ids и обратно — в Converter.

### 5.5 messages (чат)

- Список dict с ключами role, content (и опционально name, tool_calls, tool_call_id). Conjector преобразует в строку промпта по шаблону.

### 5.6 tool_calls, tool_results

- tool_calls: список {id, type: "function", function: {name, arguments}}. tool_results: список {tool_call_id, content}. Передаются между агентом и движком в agent_loop.

---

## 6. Цикл генерации и агентный цикл в движке

### 6.1 Цикл генерации (Backbone ↔ Inner Module)

Гиперграф имеет цикл: выход Inner Module (next_token_id, state) снова подаётся на вход Backbone (и при необходимости на вход Inner Module). Планировщик строит план: начальная фаза → циклическая фаза (до K раз или до stop) → конечная фаза. Перед циклом буферы заполняются от Outer Module; на каждой итерации Backbone и Inner Module выполняются; Inner Module может вернуть флаг «завершить» — тогда движок выходит из цикла и передаёт накопленную последовательность в детокенизатор. num_loop_steps (max_new_tokens) задаётся metadata или run(..., num_loop_steps=K).

### 6.2 Агентный цикл (agent_loop)

При выполнении узла-агента движок проверяет выход на наличие tool_calls. Если есть — по таблице tool_id → node_id выполняются узлы-инструменты, собираются tool_results; затем снова выполняется узел-агент с обновлёнными входами (context + tool_results). Цикл повторяется до отсутствия tool_calls или до max_steps. Детали — [03_TASK_HYPERGRAPH.md](03_TASK_HYPERGRAPH.md) §7.3, [HYPERGRAPH_ENGINE.md](HYPERGRAPH_ENGINE.md).

### 6.3 Стриминг и прерывания

При stream=True или callback движок/блок вызывает callback на каждой итерации (или на каждом новом токене) с частичным результатом. При interrupt_on движок при достижении указанного узла (или инструмента) может приостановить выполнение и вернуть управление; следующий run с теми же thread_id/checkpoint_id продолжает с сохранённого состояния.

---

## 7. Конфигурация и вызов run

### 7.1 Конфиг гиперграфа (LLM / агент)

- **nodes:** node_id, block_type, config (для каждого узла: Backbone — путь к чекпоинту, архитектура; Inner Module — temperature, top_p, stop_token_ids, процессоры; Converter — путь к токенизатору; и т.д.).
- **edges:** (source_node, source_port, target_node, target_port).
- **exposed_inputs:** (узел, порт) с именем (prompt, messages, query, и т.д.).
- **exposed_outputs:** (узел, порт) с именем (text, response, tool_calls, и т.д.).
- **metadata:** num_loop_steps, max_new_tokens, stop_token_ids, seed, interrupt_on, checkpointer_ref, tool_id_to_node_id (для агента), и др.

### 7.2 Конфиг блоков (типичные поля)

- **Backbone (LLM):** checkpoint_ref (или provider, model_id для API), vocab_size, hidden_size, num_layers, num_attention_heads, max_position_embeddings, use_cache, dtype, quantization_config.
- **Inner Module (sampler):** temperature, top_p, top_k, do_sample, repetition_penalty, no_repeat_ngram_size, min_length, min_new_tokens, eos_token_id, stop_token_ids, stop_strings; список LogitsProcessor/StoppingCriteria по типу или конфигу.
- **Converter:** путь к словарю/токенизатору, max_length, padding_side.
- **Outer Module:** prefill_chunk_size, use_cache.
- **Conjector:** chat_template (имя или шаблон), system_prompt_default.
- **Helper (RAG):** индекс, путь к корпусу, top_k. **Helper (filesystem):** backend_ref. **Helper (execute):** sandbox_ref.

### 7.3 Вызов run

`outputs = run(hypergraph, inputs, num_loop_steps=K, stream=False, **options)`

- **inputs:** словарь по exposed_inputs (prompt, messages, query, и т.д.).
- **outputs:** словарь по exposed_outputs (text, response, и т.д.).
- **options:** device, seed, max_new_tokens (переопределение metadata), stream, callback, thread_id, checkpoint_id, interrupt_on (переопределение), и др.

---

## 8. Обучение (LoRA, full fine-tune)

Обучаемые параметры принадлежат блокам. Гиперграф агрегирует state_dict по узлам; сериализация — [SERIALIZATION.md](SERIALIZATION.md). Trainable узлы задаются конфигом; обучение на уровне графа — [DOMAINS_DEPLOYMENT_TRAINING.md](DOMAINS_DEPLOYMENT_TRAINING.md) §6. LoRA (Injector) — обучаются только веса LoRA при замороженном Backbone. Full fine-tune — Backbone (и при необходимости Conjector) trainable. Дедупликация: один block_id — один state_dict в чекпоинте.

---

## 9. Сериализация

Конфиг гиперграфа (nodes, edges, exposed_inputs, exposed_outputs, metadata, schema_version) + чекпоинт (state_dict по node_id / block_id). Сохранение/загрузка по [SERIALIZATION.md](SERIALIZATION.md). Для агента с чекпоинтером сессии: опционально сохраняется пользовательское состояние (история сообщений, tool_calls) по конвенции реализации; thread_id и checkpoint_id в run options обеспечивают resume.

---

## 10. Использование на уровне воркфлоу

Цепочки: RAG-гиперграф → LLM-гиперграф → постобработка; LLM-гиперграф (описание сцены) → диффузионный гиперграф (text-to-image). num_loop_steps / max_new_tokens может задаваться по graph_id. Обучение в воркфлоу: trainable — один или несколько гиперграфов; чекпоинт воркфлоу — объединение их state_dict.

---

## 11. Граничные случаи

- **Локальный vs API:** один конфиг графа при смене Backbone с checkpoint_ref на provider/model_id не меняет контракта run.
- **Батчинг:** входы с batch-размером B; блоки обрабатывают [B, …] по портам.
- **Длина контекста:** max_length, truncation в Converter и Backbone; стратегия (слева/справа, sliding window) по конфигу.
- **Стриминг:** stream=True или callback — способ доставки результата; контракт портов тот же.
- **Устройство:** hypergraph.to(device), run(..., device=...); для API-режима device относится к локальным узлам.

---

## 12. Связь с другими документами

- [DOMAINS_DEPLOYMENT_TRAINING.md](DOMAINS_DEPLOYMENT_TRAINING.md) §4 — краткий обзор LLM; данный документ — полная спецификация и паритет.
- [02_ABSTRACT_TASK_NODES.md](02_ABSTRACT_TASK_NODES.md) — контракты портов ролей; здесь — маппинг на реализации LLM и агента.
- [03_TASK_HYPERGRAPH.md](03_TASK_HYPERGRAPH.md) §2.2, §6–7 — цикл, num_loop_steps, agent_loop.
- [HYPERGRAPH_ENGINE.md](HYPERGRAPH_ENGINE.md) — планировщик, итеративная фаза, буферы.
- [SERIALIZATION.md](SERIALIZATION.md) — конфиг + чекпоинт, дедупликация.

---

**Итог.** Документ задаёт **полную спецификацию поддержки языковых моделей и агентов** в фреймворке Иггдрасиль и **полный паритет по возможностям** с Hugging Face Transformers v5.3.0, LangChain (create_agent), LangGraph (StateGraph, Pregel), DeepAgents (create_deep_agent). Паритет достигается **полной интеграцией аналогичного функционала на базе движка гиперграфа и блоков Иггдрасиль** — без использования кода или исполнения этих библиотек как основы. Все перечисленные возможности (генерация, процессоры логитов, остановка, стриминг, кэш, пайплайн, агент с инструментами, middleware, субагенты, чекпоинт, прерывания) выражаются конфигом гиперграфа, узлами-задачами и опциями run и выполняются движком фреймворка.
