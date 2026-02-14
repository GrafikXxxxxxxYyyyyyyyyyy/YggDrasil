"""Modal.com deployment for YggDrasil.

Deploy with: modal deploy yggdrasil.deployment.cloud.modal_app
"""
from __future__ import annotations

try:
    import modal
    
    app = modal.App("yggdrasil")
    
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch",
            "diffusers",
            "transformers",
            "accelerate",
            "omegaconf",
            "tqdm",
            "pillow",
            "numpy",
        )
        .pip_install("yggdrasil")
    )
    
    @app.function(
        image=image,
        gpu="A10G",
        timeout=300,
    )
    def generate(
        model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        prompt: str = "a beautiful landscape",
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int = -1,
    ) -> bytes:
        """Generate an image and return PNG bytes."""
        import torch
        import numpy as np
        from PIL import Image
        from io import BytesIO
        
        from yggdrasil.assemblers import ModelAssembler, PipelineAssembler
        
        model = ModelAssembler.from_pretrained(model_id)
        model = model.to("cuda")
        
        sampler = PipelineAssembler.for_generation(
            model=model, num_steps=num_steps, guidance_scale=guidance_scale,
        )
        
        condition = {"text": prompt}
        codec = model._slot_children.get("codec")
        if codec and hasattr(codec, "get_latent_shape"):
            shape = codec.get_latent_shape(1, height, width)
        else:
            shape = (1, 4, height // 8, width // 8)
        
        generator = torch.Generator(device="cuda").manual_seed(seed) if seed >= 0 else None
        result = sampler.sample(condition=condition, shape=shape, generator=generator)
        
        image = result[0].cpu()
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).clip(0, 255).astype(np.uint8)
        
        buf = BytesIO()
        Image.fromarray(image).save(buf, format="PNG")
        return buf.getvalue()
    
    @app.local_entrypoint()
    def main():
        result = generate.remote(prompt="a cat sitting on a rainbow")
        with open("output.png", "wb") as f:
            f.write(result)
        print("Saved to output.png")

except ImportError:
    pass  # modal not installed
