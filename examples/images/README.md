# Image Examples

For full project structure and file descriptions, see [PROJECT_STRUCTURE.md](../../PROJECT_STRUCTURE.md) at repo root.

## Compare Diffusers vs YggDrasil (сверка)

Запуск одинакового промпта через Diffusers и YggDrasil для сравнения выхода:

```bash
# SD 1.5 (512×512)
python examples/images/compare_diffusers_yggdrasil.py --model sd15

# SDXL (1024×1024)
python examples/images/compare_diffusers_yggdrasil.py --model sdxl

# SD3 (1024×1024)
python examples/images/compare_diffusers_yggdrasil.py --model sd3

# FLUX (1024×1024)
python examples/images/compare_diffusers_yggdrasil.py --model flux

# Метрики MSE/PSNR после генерации обоих
python examples/images/compare_diffusers_yggdrasil.py --model sd15 --compare
```

Выходы: `examples/images/{sd15,sdxl,sd3,flux}/out/diffusers.png` и `yggdrasil.png`. Флаг `--compare` вычисляет MSE, PSNR (dB) и max_diff.

Для SDXL выровнены: dtype float16 на CUDA, init_noise_sigma, scale_model_input, guidance_rescale=0, randn_tensor из Diffusers. Небольшие различия могут сохраняться из‑за порядка операций с плавающей точкой.

Опции: `--prompt`, `--steps`, `--seed`, `--guidance-scale`, `--width`, `--height`, `--device`, `--diffusers-only`, `--yggdrasil-only`, `--compare`, `--out-dir`.
