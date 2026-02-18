"""T5 text conditioner for SD3, Flux, DeepFloyd, and other models."""
import torch
from typing import Dict, Any
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.model.conditioner import AbstractConditioner


@register_block("conditioner/t5_text")
class T5TextConditioner(AbstractConditioner):
    """T5 text encoder for conditioning.
    
    Used by SD3, Flux, DeepFloyd IF, and other models that need
    larger text understanding capacity than CLIP.
    """
    
    block_type = "conditioner/t5_text"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        from transformers import T5EncoderModel, T5Tokenizer
        
        pretrained = config.get("pretrained", "google/t5-v1_1-xxl")
        dtype = torch.float16 if config.get("fp16", True) else torch.float32
        
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained)
        self.text_encoder = T5EncoderModel.from_pretrained(
            pretrained, torch_dtype=dtype
        )
        self.max_length = int(config.get("max_length", 256))
        self.text_encoder.requires_grad_(False)
    
    def __call__(self, condition: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompt = condition.get("text", condition.get("prompt", ""))
        if isinstance(prompt, str):
            prompt = [prompt]
        
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.text_encoder.device)
        
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=inputs.input_ids)
            text_emb = outputs.last_hidden_state
        
        return {"text_emb": text_emb, "encoder_hidden_states": text_emb}
