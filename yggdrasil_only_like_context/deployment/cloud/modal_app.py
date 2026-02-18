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
        """Generate an image and return PNG bytes. Uses InferencePipeline (unified API)."""
        import numpy as np
        from PIL import Image
        from io import BytesIO
        
        from yggdrasil.pipeline import InferencePipeline
        
        pipe = InferencePipeline.from_pretrained(model_id, device="cuda")
        out = pipe(
            prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed if seed >= 0 else None,
        )
        if out.images and len(out.images) > 0:
            pil_img = out.images[0]
        else:
            # Fallback: raw tensor from latents/decoded
            raw = out.raw
            tensor = raw.get("decoded") or raw.get("image")
            if tensor is not None:
                if tensor.dim() == 4:
                    tensor = tensor[0]
                arr = (tensor.cpu().float() / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
                pil_img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
            else:
                raise RuntimeError("No image output from pipeline")
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()
    
    @app.local_entrypoint()
    def main():
        result = generate.remote(prompt="a cat sitting on a rainbow")
        with open("output.png", "wb") as f:
            f.write(result)
        print("Saved to output.png")

except ImportError:
    pass  # modal not installed
