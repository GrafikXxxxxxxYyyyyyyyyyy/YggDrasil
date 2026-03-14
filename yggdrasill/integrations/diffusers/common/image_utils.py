"""Image pre/post-processing utilities matching Diffusers conventions."""
from __future__ import annotations

from typing import Any, List, Optional


def _import_torch() -> Any:
    import torch
    return torch


def preprocess_image(
    image: Any,
    height: int = 512,
    width: int = 512,
    *,
    dtype: Optional[Any] = None,
    device: Optional[str] = None,
) -> Any:
    """Convert PIL/numpy/torch image to model-ready tensor [B,C,H,W] in [-1,1].

    Delegates to ``diffusers.image_processor.VaeImageProcessor`` when available,
    with a pure-tensor fallback.
    """
    torch = _import_torch()

    try:
        from diffusers.image_processor import VaeImageProcessor
        processor = VaeImageProcessor(vae_scale_factor=8)
        tensor = processor.preprocess(image, height=height, width=width)
    except ImportError:
        import numpy as np
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            image = image.resize((width, height))
            arr = np.array(image).astype(np.float32) / 255.0
            if arr.ndim == 2:
                arr = arr[:, :, None]
            arr = arr.transpose(2, 0, 1)
            tensor = torch.from_numpy(arr).unsqueeze(0)
            tensor = tensor * 2.0 - 1.0
        elif isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = image.transpose(2, 0, 1)
            tensor = torch.from_numpy(image).unsqueeze(0).float()
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            tensor = tensor * 2.0 - 1.0
        elif isinstance(image, torch.Tensor):
            tensor = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def preprocess_mask(
    mask: Any,
    height: int = 512,
    width: int = 512,
    *,
    dtype: Optional[Any] = None,
    device: Optional[str] = None,
) -> Any:
    """Convert mask to tensor [B,1,H/8,W/8] in [0,1]."""
    torch = _import_torch()

    try:
        from diffusers.image_processor import VaeImageProcessor
        processor = VaeImageProcessor(
            vae_scale_factor=8,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        tensor = processor.preprocess(mask, height=height, width=width)
    except ImportError:
        import numpy as np
        from PIL import Image as PILImage

        if isinstance(mask, PILImage.Image):
            mask = mask.convert("L").resize((width // 8, height // 8))
            arr = np.array(mask).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        elif isinstance(mask, np.ndarray):
            tensor = torch.from_numpy(mask).float()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif isinstance(mask, torch.Tensor):
            tensor = mask
        else:
            raise TypeError(f"Unsupported mask type: {type(mask)}")

    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def postprocess_image(
    images: Any,
    output_type: str = "pil",
) -> Any:
    """Convert model output tensor to the requested format.

    Accepts tensors in [0,1] range with shape [B,C,H,W].
    """
    torch = _import_torch()
    import numpy as np

    if output_type == "latent":
        return images

    if isinstance(images, torch.Tensor):
        images = images.clamp(0, 1).cpu().float()
        images_np = images.permute(0, 2, 3, 1).numpy()
    elif isinstance(images, np.ndarray):
        images_np = images
    else:
        return images

    if output_type == "np":
        return images_np

    if output_type == "pt":
        return torch.from_numpy(images_np).permute(0, 3, 1, 2)

    return numpy_to_pil(images_np)


def numpy_to_pil(images: Any) -> List[Any]:
    """Convert numpy [B,H,W,C] float32 in [0,1] to list of PIL images."""
    import numpy as np
    from PIL import Image as PILImage

    if images.ndim == 3:
        images = images[np.newaxis, ...]
    images = (images * 255).round().astype(np.uint8)
    return [PILImage.fromarray(img) for img in images]
