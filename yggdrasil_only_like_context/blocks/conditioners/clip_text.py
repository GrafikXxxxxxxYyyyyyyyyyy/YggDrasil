import torch
from omegaconf import DictConfig
from transformers import CLIPTextModel, CLIPTokenizer

from ...core.block.registry import register_block
from ...core.model.conditioner import AbstractConditioner


@register_block("conditioner/clip_text")
class CLIPTextConditioner(AbstractConditioner):
    """CLIPTextModel + Tokenizer для SD 1.5.
    
    Текстовый энкодер ВСЕГДА загружается в float32, потому что
    LayerNorm не поддерживает float16 на MPS, а размер модели
    небольшой (~490 МБ). Выход (embedding) кастится к dtype
    backbone автоматически.
    """
    
    block_type = "conditioner/clip_text"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        pretrained = config.get("pretrained", "openai/clip-vit-large-patch14")
        tokenizer_sub = config.get("tokenizer_subfolder")
        text_encoder_sub = config.get("text_encoder_subfolder")
        
        kw_t = {}
        # Text encoder always float32 — LayerNorm crashes in float16 on MPS
        kw_m = {"torch_dtype": torch.float32}
        if tokenizer_sub is not None:
            kw_t["subfolder"] = tokenizer_sub
        if text_encoder_sub is not None:
            kw_m["subfolder"] = text_encoder_sub
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained, **kw_t)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained, **kw_m)
        self.text_encoder.requires_grad_(False)
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
        
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(inputs.input_ids)[0]
        
        return {"encoder_hidden_states": encoder_hidden_states}