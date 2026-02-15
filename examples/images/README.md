# Image Examples

## Compare Diffusers vs YggDrasil

Run the same prompt through both Diffusers and YggDrasil to compare output quality:

```bash
# SD 1.5 (512×512)
python examples/images/compare_diffusers_yggdrasil.py --model sd15

# SDXL (1024×1024)
python examples/images/compare_diffusers_yggdrasil.py --model sdxl

# SD3 (1024×1024)
python examples/images/compare_diffusers_yggdrasil.py --model sd3
```

Outputs: `examples/images/{sd15,sdxl,sd3}/out/diffusers.png` and `yggdrasil.png`.

Or run from a model folder (delegates to the shared script):

```bash
python examples/images/sd15/compare_diffusers_yggdrasil.py
```

Options: `--prompt`, `--steps`, `--seed`, `--guidance-scale`, `--width`, `--height`, `--device`, `--diffusers-only`, `--yggdrasil-only`, `--out-dir`.
