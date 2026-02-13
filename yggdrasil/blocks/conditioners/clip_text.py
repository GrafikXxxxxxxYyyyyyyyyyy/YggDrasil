import torch
from omegaconf import DictConfig
from transformers import CLIPTextModel, CLIPTokenizer

from ...core.block.registry import register_block
from ...core.model.conditioner import AbstractConditioner


@register_block("conditioner/clip_text")
class CLIPTextConditioner(AbstractConditioner):
    """CLIPTextModel + Tokenizer для SD 1.5."""
    
    block_type = "conditioner/clip_text"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            config.get("pretrained", "openai/clip-vit-large-patch14"),
            torch_dtype=torch.float16
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            config.get("pretrained", "openai/clip-vit-large-patch14"),
            torch_dtype=torch.float16
        )
        self.max_length = config.get("max_length", 77)
    
    def __call__(self, condition: dict) -> dict:
        prompt = condition.get("text", "")
        if isinstance(prompt, str):
            prompt = [prompt]
        
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.text_encoder.device)
        
        encoder_hidden_states = self.text_encoder(inputs.input_ids)[0]
        
        return {"encoder_hidden_states": encoder_hidden_states}