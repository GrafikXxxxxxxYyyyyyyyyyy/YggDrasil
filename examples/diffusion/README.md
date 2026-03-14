# Diffusion Examples

This folder contains practical examples for the current diffusion layer.

## SDXL

- `sdxl_text2img.py` - build an SDXL text-to-image graph via `Hypergraph.from_template(...)`.
- `sdxl_img2img.py` - build an SDXL image-to-image graph via `Hypergraph.from_template(...)`.
- `sdxl_base_refiner.py` - build the SDXL base/refiner workflow via `Workflow.from_template(...)`.
- `sdxl_manual_graph.py` - load components manually and assemble an SDXL graph.

## Notes

- Install diffusion dependencies first:

```bash
pip install -e ".[diffusion]"
```

- These examples are focused on the current graph-first SDXL API:
  template construction, component loading, validation, exposed ports,
  and config serialization.
- They intentionally print validation results before execution so you can
  inspect the generated structure while the runtime loop wiring is being
  finalized.
