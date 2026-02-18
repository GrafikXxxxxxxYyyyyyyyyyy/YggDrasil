# yggdrasil/blocks/conditioners/sd3_text.py
"""SD3 text conditioner: CLIP (ViT-L) + CLIP (bigG) + T5-XXL, combined as in Diffusers StableDiffusion3Pipeline.

Produces encoder_hidden_states (B, 77+256, 4096) and pooled_embedding (B, 768+1280) for SD3Transformer2DModel.
"""
from __future__ import annotations

import torch
from typing import Dict, Any, List
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.model.conditioner import AbstractConditioner


def _get_clip_embeds(
    tokenizer,
    text_encoder,
    prompt: List[str],
    device: torch.device,
    max_length: int = 77,
    clip_skip: int | None = None,
):
    """Get prompt_embeds (penultimate hidden) and pooled from one CLIP. Same as Diffusers _get_clip_prompt_embeds."""
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    with torch.no_grad():
        outputs = text_encoder(text_input_ids, output_hidden_states=True)
    pooled = outputs[0]  # pooler_output / projection
    if clip_skip is None:
        prompt_embeds = outputs.hidden_states[-2]
    else:
        prompt_embeds = outputs.hidden_states[-(clip_skip + 2)]
    if pooled is None:
        pooled = prompt_embeds[:, 0]
    return prompt_embeds.to(device), pooled.to(device)


def _get_t5_embeds(tokenizer_3, text_encoder_3, prompt: List[str], device: torch.device, max_length: int = 256):
    """Get T5 encoder hidden state. Same as Diffusers _get_t5_prompt_embeds."""
    text_inputs = tokenizer_3(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    with torch.no_grad():
        prompt_embeds = text_encoder_3(input_ids)[0]
    return prompt_embeds.to(device)


@register_block("conditioner/sd3_text")
class SD3TextConditioner(AbstractConditioner):
    """SD3 triple text encoder: CLIP ViT-L + CLIP bigG + T5-XXL.

    Loads from pretrained repo subfolders (text_encoder, text_encoder_2, text_encoder_3, tokenizer, tokenizer_2, tokenizer_3).
    Outputs encoder_hidden_states (B, 333, 4096) and pooled_embedding (B, 2048) for SD3Transformer2DModel.
    """

    block_type = "conditioner/sd3_text"

    def __init__(self, config: DictConfig):
        super().__init__(config)
        from transformers import (
            CLIPTextModelWithProjection,
            CLIPTokenizer,
            T5EncoderModel,
            T5TokenizerFast,
        )
        pretrained = config.get("pretrained", "stabilityai/stable-diffusion-3-medium-diffusers")
        dtype = torch.float32  # encoders often kept in float32 for stability
        if config.get("fp16"):
            dtype = torch.float16

        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained, subfolder="tokenizer")
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            pretrained, subfolder="text_encoder", torch_dtype=dtype
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained, subfolder="tokenizer_2")
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            pretrained, subfolder="text_encoder_2", torch_dtype=dtype
        )
        self.tokenizer_3 = T5TokenizerFast.from_pretrained(pretrained, subfolder="tokenizer_3")
        self.text_encoder_3 = T5EncoderModel.from_pretrained(
            pretrained, subfolder="text_encoder_3", torch_dtype=dtype
        )
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.text_encoder_3.requires_grad_(False)
        self.max_length = int(config.get("max_length", 77))
        self.t5_max_length = int(config.get("t5_max_length", 256))

    def __call__(self, condition: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompt = condition.get("text", condition.get("prompt", ""))
        if isinstance(prompt, str):
            prompt = [prompt]
        device = next(self.text_encoder.parameters()).device

        # CLIP 1 (ViT-L) and CLIP 2 (bigG)
        prompt_embed, pooled_embed = _get_clip_embeds(
            self.tokenizer, self.text_encoder, prompt, device, self.max_length
        )
        prompt_2_embed, pooled_2_embed = _get_clip_embeds(
            self.tokenizer_2, self.text_encoder_2, prompt, device, self.max_length
        )
        # (B, 77, 768) and (B, 77, 1280) -> (B, 77, 2048)
        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
        # Pad to T5 dim 4096
        t5_embed = _get_t5_embeds(
            self.tokenizer_3, self.text_encoder_3, prompt, device, self.t5_max_length
        )
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )
        # (B, 77, 4096) and (B, 256, 4096) -> (B, 333, 4096)
        encoder_hidden_states = torch.cat([clip_prompt_embeds, t5_embed], dim=-2)
        pooled_embedding = torch.cat([pooled_embed, pooled_2_embed], dim=-1)

        return {
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_embedding": pooled_embedding,
            "embedding": {
                "encoder_hidden_states": encoder_hidden_states,
                "pooled_embedding": pooled_embedding,
            },
        }

    def process(self, **port_inputs) -> dict:
        raw = port_inputs.get("raw_condition", {})
        if not isinstance(raw, dict):
            raw = {"text": str(raw)}
        result = self(raw)
        # AbstractConditioner expects "embedding" for the loop; pass full dict so backbone gets both keys
        out = dict(result)
        out["embedding"] = result["embedding"]
        return out
