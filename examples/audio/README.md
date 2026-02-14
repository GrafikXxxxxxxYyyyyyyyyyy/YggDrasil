# Аудио — примеры пайплайнов генерации звука

Примеры использования единой блочной системы генерации аудио в YggDrasil: text-to-audio по шаблонам и по HuggingFace model ID.

---

## Шаблоны (Lego-пайплайны)

| Шаблон | Описание |
|--------|----------|
| `musicldm_txt2audio` | MusicLDM — текст в музыку (ucsd-reach/musicldm), conditioner: CLAP |
| `audioldm_txt2audio` | AudioLDM — текст в звук (cvssp/audioldm-l-full) |
| `audioldm2_txt2audio` | AudioLDM 2 — текст в звук (cvssp/audioldm2) |
| `stable_audio` | Stable Audio — текст в звук (Stability AI) |
| `dance_diffusion_audio` | Dance Diffusion (harmonai/maestro-150k) |
| `aura_flow_audio` | Aura Flow — голос/TTS (black-forest-labs/AuraFlow) |

---

## Примеры

### generate.py — базовая генерация по шаблону

```bash
cd /path/to/YggDrasil
python examples/audio/generate.py
python examples/audio/generate.py --template audioldm2_txt2audio --prompt "rain and thunder" --steps 100
python examples/audio/generate.py --template musicldm_txt2audio --output music.wav --seed 123
```

Параметры: `--template`, `--prompt`, `--negative_prompt`, `--steps`, `--guidance_scale`, `--seed`, `--output`, `--sample_rate`, `--device`.

### from_pretrained.py — загрузка по HuggingFace model ID

```bash
python examples/audio/from_pretrained.py
python examples/audio/from_pretrained.py --model_id cvssp/audioldm2 --prompt "rain and thunder"
python examples/audio/from_pretrained.py --model_id ucsd-reach/musicldm --output music.wav
```

Автоматически выбирается шаблон по имени репозитория (audioldm2 → audioldm2_txt2audio, musicldm → musicldm_txt2audio и т.д.).

### list_templates.py — список аудио-шаблонов

```bash
python examples/audio/list_templates.py
python examples/audio/list_templates.py --all
```

---

## Требования

- PyTorch, YggDrasil, зависимости из `requirements.txt`
- Для WAV: `scipy` (рекомендуется) или только стандартная библиотека (`wave`)
- CUDA рекомендуется; на CPU генерация может быть очень медленной

---

## Sample rate

Частота дискретизации выхода зависит от модели (например, 16 kHz для AudioLDM/MusicLDM). Параметр `--sample_rate` при сохранении WAV можно подобрать под конкретную модель (16000, 22050, 44100).
