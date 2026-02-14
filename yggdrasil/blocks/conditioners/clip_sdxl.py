# yggdrasil/blocks/conditioners/clip_sdxl.py
"""SDXL dual text encoder: CLIP L (768) + CLIP G with projection (1280), concat + pooled for added_cond_kwargs."""
from __future__ import annotations

import torch
from omegaconf import DictConfig
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from ...core.block.registry import register_block
from ...core.block.port import Port, InputPort, OutputPort, TensorSpec
from ...core.model.conditioner import AbstractConditioner


def _encode_prompt_sdxl(
    tokenizer,
    tokenizer_2,
    text_encoder,
    text_encoder_2,
    prompt: str | list,
    device: torch.device,
    num_images_per_prompt: int = 1,
    negative_prompt: str | list | None = None,
    force_zeros_for_empty_prompt: bool = True,
    max_length_1: int = 77,
    max_length_2: int = 77,
):
    """Encode prompt with both encoders; return (prompt_embeds, pooled_embeds) and (uncond_embeds, uncond_pooled)."""
    prompt = [prompt] if isinstance(prompt, str) else list(prompt)
    batch_size = len(prompt)
    prompt_2 = prompt  # SDXL uses same prompt for both encoders by default

    tokenizers = [tokenizer, tokenizer_2] if tokenizer is not None else [tokenizer_2]
    text_encoders = [text_encoder, text_encoder_2] if text_encoder is not None else [text_encoder_2]

    pooled = None
    prompt_embeds_list = []
    for p, tok, enc in zip([prompt, prompt_2], tokenizers, text_encoders):
        text_inputs = tok(
            p,
            padding="max_length",
            max_length=tok.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        with torch.no_grad():
            out = enc(text_input_ids, output_hidden_states=True)
        # Penultimate layer for text; pooled only from final encoder (CLIP G)
        h = out.hidden_states[-2]
        prompt_embeds_list.append(h)
        if enc == text_encoder_2:
            pooled = getattr(out, "pooler_output", None)
            if pooled is None and hasattr(out, "__getitem__") and len(out) and getattr(out[0], "ndim", 0) == 2:
                pooled = out[0]
            if pooled is None and hasattr(out, "last_hidden_state"):
                pooled = out.last_hidden_state[:, 0]
    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
    if pooled is None:
        pooled = prompt_embeds[:, 0]
    pooled_prompt_embeds = pooled

    # Unconditional
    zero_out = force_zeros_for_empty_prompt and (negative_prompt is None or (isinstance(negative_prompt, str) and negative_prompt.strip() == ""))
    if zero_out:
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled = torch.zeros_like(pooled_prompt_embeds)
    else:
        neg = negative_prompt or ""
        neg = [neg] if isinstance(neg, str) else list(neg)
        neg_2 = neg
        negative_prompt_embeds_list = []
        neg_pooled = None
        for n, tok, enc in zip([neg, neg_2], tokenizers, text_encoders):
            max_len = prompt_embeds.shape[1]
            uncond_input = tok(n, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt")
            with torch.no_grad():
                out = enc(uncond_input.input_ids.to(device), output_hidden_states=True)
            negative_prompt_embeds_list.append(out.hidden_states[-2])
            if enc == text_encoder_2:
                neg_pooled = getattr(out, "pooler_output", None)
                if neg_pooled is None and hasattr(out, "__getitem__") and len(out) and getattr(out[0], "ndim", 0) == 2:
                    neg_pooled = out[0]
                if neg_pooled is None and hasattr(out, "last_hidden_state"):
                    neg_pooled = out.last_hidden_state[:, 0]
        negative_prompt_embeds = torch.cat(negative_prompt_embeds_list, dim=-1)
        negative_pooled = neg_pooled if neg_pooled is not None else negative_prompt_embeds[:, 0]

    # Repeat for num_images_per_prompt
    bs, seq, dim = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(bs * num_images_per_prompt, seq, dim)
    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(bs * num_images_per_prompt, -1)
    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1).view(bs * num_images_per_prompt, seq, dim)
    negative_pooled = negative_pooled.repeat(1, num_images_per_prompt).view(bs * num_images_per_prompt, -1)

    return (
        (prompt_embeds, pooled_prompt_embeds),
        (negative_prompt_embeds, negative_pooled),
    )


def _get_add_time_ids(height: int, width: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Original size = crop = target = (height, width), crop top-left = (0, 0)."""
    add_time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]],
        device=device,
        dtype=dtype,
    ).repeat(batch_size, 1)
    return add_time_ids


@register_block("conditioner/clip_sdxl")
class CLIPSDXLConditioner(AbstractConditioner):
    """Dual text encoder for SDXL: CLIP L + CLIP G, outputs condition/uncond dicts with encoder_hidden_states and added_cond_kwargs (text_embeds, time_ids)."""

    block_type = "conditioner/clip_sdxl"

    @classmethod
    def declare_io(cls) -> dict:
        return {
            "prompt": InputPort("prompt", data_type="any", description="Prompt text"),
            "negative_prompt": InputPort("negative_prompt", data_type="any", optional=True, description="Negative prompt"),
            "height": InputPort("height", data_type="scalar", optional=True, description="Target height for time_ids"),
            "width": InputPort("width", data_type="scalar", optional=True, description="Target width for time_ids"),
            "batch_size": InputPort("batch_size", data_type="scalar", optional=True, description="Batch size for time_ids"),
            "num_images_per_prompt": InputPort("num_images_per_prompt", data_type="scalar", optional=True, description="Images per prompt"),
            "condition": OutputPort("condition", data_type="dict", description="Condition dict: encoder_hidden_states, added_cond_kwargs"),
            "uncond": OutputPort("uncond", data_type="dict", description="Uncond dict: encoder_hidden_states, added_cond_kwargs"),
            "embedding": OutputPort("embedding", data_type="dict", description="Alias for condition (for graph compatibility)"),
        }

    def __init__(self, config: DictConfig):
        super().__init__(config)
        pretrained = config.get("pretrained", "stabilityai/stable-diffusion-xl-base-1.0")
        self.force_zeros_for_empty_prompt = config.get("force_zeros_for_empty_prompt", True)
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained, subfolder="tokenizer_2")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained, subfolder="text_encoder", torch_dtype=torch.float32)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained, subfolder="text_encoder_2", torch_dtype=torch.float32)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)

    def process(self, **port_inputs) -> dict:
        prompt = port_inputs.get("prompt", port_inputs.get("raw_condition", {}).get("text", ""))
        if isinstance(prompt, dict):
            prompt = prompt.get("text", "")
        negative_prompt = port_inputs.get("negative_prompt", None)
        height = int(port_inputs.get("height", 1024))
        width = int(port_inputs.get("width", 1024))
        batch_size = int(port_inputs.get("batch_size", 1))
        num_images_per_prompt = int(port_inputs.get("num_images_per_prompt", 1))
        device = next(self.text_encoder_2.parameters()).device
        dtype_emb = self.text_encoder_2.dtype

        (prompt_embeds, pooled_prompt_embeds), (negative_prompt_embeds, negative_pooled) = _encode_prompt_sdxl(
            self.tokenizer,
            self.tokenizer_2,
            self.text_encoder,
            self.text_encoder_2,
            prompt,
            device,
            num_images_per_prompt=num_images_per_prompt,
            negative_prompt=negative_prompt,
            force_zeros_for_empty_prompt=self.force_zeros_for_empty_prompt,
        )
        effective_batch = batch_size * num_images_per_prompt
        add_time_ids = _get_add_time_ids(height, width, effective_batch, device, dtype_emb)

        condition = {
            "encoder_hidden_states": prompt_embeds.to(dtype_emb),
            "added_cond_kwargs": {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            },
        }
        uncond = {
            "encoder_hidden_states": negative_prompt_embeds.to(dtype_emb),
            "added_cond_kwargs": {
                "text_embeds": negative_pooled.to(dtype_emb),
                "time_ids": add_time_ids,
            },
        }
        return {
            "condition": condition,
            "uncond": uncond,
            "embedding": condition,
        }
