"""CLIP Image Encoder conditioner for image-conditioned generation."""
import torch
from typing import Dict, Any
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.model.conditioner import AbstractConditioner


@register_block("conditioner/image_encoder")
class ImageEncoderConditioner(AbstractConditioner):
    """CLIP Image encoder for image conditioning.
    
    Used for img2img, IP-Adapter, and style transfer.
    Encodes reference images into CLIP embedding space.
    """
    
    block_type = "conditioner/image_encoder"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._model = None
        self._processor = None
        self.pretrained = config.get("pretrained", "openai/clip-vit-large-patch14")
    
    def _load(self):
        if self._model is not None:
            return
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
        
        self._processor = CLIPImageProcessor.from_pretrained(self.pretrained)
        self._model = CLIPVisionModelWithProjection.from_pretrained(
            self.pretrained, torch_dtype=torch.float16, low_cpu_mem_usage=False
        )
        self._model.eval()
    
    def __call__(self, condition: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        image = condition.get("image", condition.get("reference_image"))
        if image is None:
            return {}
        
        self._load()
        
        # Process image
        if not isinstance(image, torch.Tensor):
            inputs = self._processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(
                device=next(self._model.parameters()).device,
                dtype=torch.float16,
            )
        else:
            pixel_values = image
        
        with torch.no_grad():
            outputs = self._model(pixel_values=pixel_values)
            image_embeds = outputs.image_embeds
        
        return {
            "image_emb": image_embeds,
            "image_hidden_states": outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state") else image_embeds.unsqueeze(1),
        }
