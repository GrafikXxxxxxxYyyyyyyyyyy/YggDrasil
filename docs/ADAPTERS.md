# Подключение адаптеров к любому графу

Любой адаптер (ControlNet, IP-Adapter и т.д.) можно добавить к графу с `denoise_loop` и backbone, поддерживающим `adapter_features` или инъекцию.

## Универсальный API

```python
from yggdrasil.core.graph import ComputeGraph, add_controlnet_to_graph, add_ip_adapter_to_graph, add_adapter_to_graph
from yggdrasil.pipeline import Pipeline

# Любой граф: SD 1.5, SDXL, и т.д.
graph = ComputeGraph.from_template("sdxl_txt2img")
# Добавить ControlNet — появится вход control_image
graph.with_adapter("controlnet", controlnet_pretrained="xinsir/controlnet-sdxl-1.0-canny")
# или явно:
add_controlnet_to_graph(graph, controlnet_pretrained="lllyasviel/control_v11p_sd15_canny")

pipe = Pipeline(graph=graph)
out = pipe("a cat", control_image=my_canny_tensor, num_steps=30)
```

## Передача данных по ссылке

Вместо нового входа графа можно подвести вход адаптера с выхода другого узла **(node_name, port_name)**:

```python
# Контрольное изображение с выхода препроцессора (по ссылке)
graph.add_node("canny", canny_block)
graph.expose_input("source_image", "canny", "image")  # или connect от другого узла
add_controlnet_to_graph(graph, control_image_source=("canny", "output"))
# Теперь control_image = результат работы узла "canny" порт "output"
```

Аналогично для IP-Adapter: `image_source=("node_name", "port_name")`.

## Методы графа

- **graph.with_adapter(adapter_type, input_source=None, **kwargs)** — универсальное добавление адаптера.
- **graph.with_controlnet(..., control_image_source=None)** — то же для ControlNet с явным параметром.

## Передача изображения по URL или пути

Входы `control_image`, `ip_image`, `source_image` в `pipe(...)` можно передавать строкой:

- **URL:** `pipe(control_image="https://example.com/canny.png", ...)` — изображение будет загружено по HTTP(S).
- **Путь к файлу:** `pipe(control_image="/path/to/image.png", ...)`.

Загрузка выполняется автоматически внутри `Pipeline.__call__`. Утилита `load_image_from_url_or_path(source)` доступна в `yggdrasil.pipeline` для использования в своих скриптах.

## Поддерживаемые графы

- SD 1.5 (txt2img, img2img, txt2img_nobatch)
- SDXL (txt2img с batched CFG — ControlNet поддерживается)
- Любой граф с узлом `denoise_loop` и backbone с портом `adapter_features`

## ControlNet для SDXL

Используйте претрейн под SDXL, например:

- **По глубине (depth):** `diffusers/controlnet-depth-sdxl-1.0` — можно подавать обычное фото или готовую depth-карту.
- **Canny (рёбра):** `xinsir/controlnet-sdxl-1.0-canny`
- Или любой другой HF model ID для ControlNet SDXL
