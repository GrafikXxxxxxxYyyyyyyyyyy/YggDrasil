"""Build and inspect an SDXL text-to-image graph."""
from __future__ import annotations

from pathlib import Path
from pprint import pprint

from yggdrasill.engine.structure import Hypergraph
from yggdrasill.engine.validator import validate
from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes


def main() -> None:
    register_diffusion_nodes()

    graph = Hypergraph.from_template(
        "sdxl_text2img",
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        device="cuda",
        torch_dtype="float16",
        config={
            "height": 1024,
            "width": 1024,
            "guidance_scale": 7.5,
            "guidance_rescale": 0.0,
            "num_inference_steps": 30,
            "output_type": "pil",
            "seed": 42,
        },
    )

    result = validate(graph)

    print(f"graph_id: {graph.graph_id}")
    print(f"nodes: {sorted(graph.node_ids)}")
    print("input_spec:")
    pprint(graph.get_input_spec(include_dtype=True))
    print("output_spec:")
    pprint(graph.get_output_spec(include_dtype=True))
    print(f"valid: {result.valid}")
    if result.errors:
        print("validation_errors:")
        pprint(result.errors)
    if result.warnings:
        print("validation_warnings:")
        pprint(result.warnings)

    out_dir = Path("artifacts/sdxl_text2img")
    graph.save_config(out_dir, filename="config.json")
    print(f"saved config to: {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
