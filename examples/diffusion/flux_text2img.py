"""FLUX.1 text-to-image generation using YggDrasill.

Usage:
    pip install -e ".[diffusion]"
    python examples/diffusion/flux_text2img.py

Requires: ~24 GB VRAM for FLUX.1-dev in bfloat16.
For lower memory, use FLUX.1-schnell or enable CPU offloading.
"""
from yggdrasill.engine.structure import Hypergraph

graph = Hypergraph.from_template(
    "flux_text2img",
    repo_id="black-forest-labs/FLUX.1-dev",
    torch_dtype="bfloat16",
    device="cuda",
)

result = graph.run({
    "prompt": "A majestic castle floating in the clouds, cinematic lighting",
    "guidance": 3.5,
})

image = result["output_image"]
if hasattr(image, "save"):
    image.save("flux_text2img_output.png")
    print("Saved: flux_text2img_output.png")
elif isinstance(image, list) and len(image) > 0:
    image[0].save("flux_text2img_output.png")
    print("Saved: flux_text2img_output.png")
