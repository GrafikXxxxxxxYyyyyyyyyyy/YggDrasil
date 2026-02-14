"""Mistral3 conditioner for FLUX.2.

FLUX.2 uses Mistral3ForConditionalGeneration as text encoder.
Extracts hidden states from layers [10, 20, 30] and stacks them
to produce joint_attention_dim = 3 * hidden_dim = 15360.

Based on diffusers v0.36.0 Flux2Pipeline._get_mistral_3_small_prompt_embeds.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.block.port import InputPort, OutputPort, TensorSpec
from ...core.model.conditioner import AbstractConditioner


@register_block("conditioner/mistral3")
class Mistral3Conditioner(AbstractConditioner):
    """Mistral3 text encoder for FLUX.2.

    Extracts multi-layer hidden states and stacks them.
    joint_attention_dim = len(hidden_layers) * model_hidden_dim

    Config:
        pretrained: str (e.g. 'mistralai/Mistral-Small-3.1-24B-Instruct-2503')
        max_length: int (default 512)
        hidden_layers: list[int] (default [10, 20, 30])
    """

    block_type = "conditioner/mistral3"

    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "conditioner/mistral3"}
        super().__init__(config)
        self._model = None
        self._processor = None
        self.pretrained = self.config.get("pretrained", "mistralai/Mistral-Small-3.1-24B-Instruct-2503")
        self.max_length = int(self.config.get("max_length", 512))
        self.hidden_layers = list(self.config.get("hidden_layers", [10, 20, 30]))
        self.embedding_dim = int(self.config.get("embedding_dim", 15360))
        self._build_model()

    @classmethod
    def declare_io(cls) -> dict:
        return {
            "raw_condition": InputPort("raw_condition", data_type="dict",
                                       description="Dict with 'text' key"),
            "embedding": OutputPort("embedding", spec=TensorSpec(space="embedding"),
                                    description="Multi-layer hidden states [B, L, joint_dim]"),
            "text_ids": OutputPort("text_ids", data_type="tensor",
                                   description="Text position IDs for RoPE"),
        }

    def _build_model(self):
        try:
            from transformers import Mistral3ForConditionalGeneration, AutoProcessor
            self._model = Mistral3ForConditionalGeneration.from_pretrained(
                self.pretrained, torch_dtype=torch.bfloat16,
            )
            self._processor = AutoProcessor.from_pretrained(self.pretrained)
            self._model.requires_grad_(False)
        except Exception:
            self._model = None

    def process(self, **port_inputs) -> dict:
        raw = port_inputs.get("raw_condition", {})
        if not isinstance(raw, dict):
            raw = {"text": str(raw)}

        text = raw.get("text", "")

        if self._model is not None and self._processor is not None:
            with torch.no_grad():
                prompt = [text] if isinstance(text, str) else text
                device = next(self._model.parameters()).device

                # Format as chat messages
                messages = [[
                    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                    {"role": "user", "content": [{"type": "text", "text": p}]},
                ] for p in prompt]

                inputs = self._processor.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                )

                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                output = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

                # Stack hidden states from specified layers
                out = torch.stack(
                    [output.hidden_states[k] for k in self.hidden_layers], dim=1
                )
                B, num_ch, seq_len, hdim = out.shape
                prompt_embeds = out.permute(0, 2, 1, 3).reshape(B, seq_len, num_ch * hdim)

                return {
                    "embedding": prompt_embeds,
                    "encoder_hidden_states": prompt_embeds,
                    "output": prompt_embeds,
                }

        # Stub: zero embeddings
        emb = torch.zeros(1, self.max_length, self.embedding_dim)
        return {
            "embedding": emb,
            "encoder_hidden_states": emb,
            "output": emb,
        }

    def __call__(self, condition):
        return self.process(raw_condition=condition)
