"""Build and inspect the SDXL base/refiner workflow."""
from __future__ import annotations

from pathlib import Path
from pprint import pprint

from yggdrasill.engine.validator import validate
from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes
from yggdrasill.workflow.workflow import Workflow


def main() -> None:
    register_diffusion_nodes()

    workflow = Workflow.from_template(
        "sdxl_base_refiner",
        base_repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        refiner_repo_id="stabilityai/stable-diffusion-xl-refiner-1.0",
        device="cuda",
        torch_dtype="float16",
        denoising_end=0.8,
        config={
            "height": 1024,
            "width": 1024,
            "guidance_scale": 7.0,
            "num_inference_steps": 40,
        },
    )

    result = validate(workflow)

    print(f"workflow_id: {workflow.workflow_id}")
    print(f"graphs: {sorted(workflow.node_ids)}")
    print("input_spec:")
    pprint(workflow.get_input_spec(include_dtype=True))
    print("output_spec:")
    pprint(workflow.get_output_spec(include_dtype=True))
    print(f"valid: {result.valid}")
    if result.errors:
        print("validation_errors:")
        pprint(result.errors)

    out_dir = Path("artifacts/sdxl_base_refiner")
    workflow.save_config(out_dir, filename="config.json")
    print(f"saved config to: {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
