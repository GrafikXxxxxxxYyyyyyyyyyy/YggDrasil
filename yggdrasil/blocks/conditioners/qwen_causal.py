"""Qwen (Qwen2/Qwen3) causal LM conditioner for FLUX.2 [klein].

Klein uses Qwen as text encoder from the model repo (subfolder "text_encoder").
joint_attention_dim = 7680 = 3 * hidden_size (e.g. 3 * 2560).
Extracts hidden states from 3 layers and concatenates them.

Ref: diffusers Flux2KleinPipeline, black-forest-labs/FLUX.2-klein-4B.
"""
import torch
from typing import Dict, Any
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.block.port import InputPort, OutputPort, TensorSpec
from ...core.model.conditioner import AbstractConditioner


@register_block("conditioner/qwen_causal")
class QwenCausalConditioner(AbstractConditioner):
    """Qwen2/Qwen3 text encoder for FLUX.2 [klein].

    Loads from pretrained (optionally subfolder "text_encoder" for Klein repo).
    joint_attention_dim = len(hidden_layers) * model_hidden_dim (e.g. 3 * 2560 = 7680).

    Config:
        pretrained: str (e.g. 'black-forest-labs/FLUX.2-klein-4B' with subfolder 'text_encoder')
        subfolder: str (default 'text_encoder' for Klein)
        max_length: int (default 512)
        hidden_layers: list[int] (default [10, 20, 30] — indices for 3 layers → 7680 if hidden 2560)
        embedding_dim: int (default 7680)
    """

    block_type = "conditioner/qwen_causal"

    def __init__(self, config: DictConfig | dict = None):
        config = config or {"type": "conditioner/qwen_causal"}
        super().__init__(config)
        self._model = None
        self._processor = None
        self.pretrained = self.config.get("pretrained", "black-forest-labs/FLUX.2-klein-4B")
        self.subfolder = self.config.get("subfolder", "text_encoder")
        self.max_length = int(self.config.get("max_length", 512))
        self.hidden_layers = list(self.config.get("hidden_layers", [10, 20, 30]))
        self.embedding_dim = int(self.config.get("embedding_dim", 7680))
        self._build_model()

    @classmethod
    def declare_io(cls) -> dict:
        return {
            "raw_condition": InputPort("raw_condition", data_type="dict",
                                       description="Dict with 'text' key"),
            "embedding": OutputPort("embedding", spec=TensorSpec(space="embedding"),
                                    description="Multi-layer hidden states [B, L, joint_dim]"),
        }

    def _build_model(self):
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            load_kwargs = {"torch_dtype": torch.bfloat16}
            if self.config.get("token") is not None:
                load_kwargs["token"] = self.config.get("token")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.pretrained,
                subfolder=self.subfolder if self.subfolder else None,
                **load_kwargs,
            )
            proc_kw = {} if not self.config.get("token") else {"token": self.config.get("token")}
            self._processor = AutoProcessor.from_pretrained(
                self.pretrained,
                subfolder=self.subfolder if self.subfolder else None,
                **proc_kw,
            )
            self._model.requires_grad_(False)
        except Exception:
            self._model = None
            self._processor = None

    def process(self, **port_inputs) -> dict:
        raw = port_inputs.get("raw_condition", {})
        if not isinstance(raw, dict):
            raw = {"text": str(raw)}
        text = raw.get("text", "")

        if self._model is not None and self._processor is not None:
            with torch.no_grad():
                prompt = [text] if isinstance(text, str) else text
                device = next(self._model.parameters()).device
                tokenizer = getattr(self._processor, "tokenizer", self._processor)
                # Klein repo tokenizer may not have chat_template — tokenize raw text
                try:
                    if getattr(tokenizer, "chat_template", None) is not None:
                        messages = [[{"role": "user", "content": p}] for p in prompt]
                        text_token = self._processor.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=False,
                            return_dict=True,
                            return_tensors="pt",
                            padding="max_length",
                            truncation=True,
                            max_length=self.max_length,
                        )
                    else:
                        raise AttributeError("no chat_template")
                except (ValueError, AttributeError, TypeError):
                    text_token = tokenizer(
                        prompt,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                input_ids = text_token["input_ids"].to(device)
                attention_mask = text_token.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                output = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                out = torch.stack(
                    [output.hidden_states[k] for k in self.hidden_layers], dim=1
                )
                B, num_ch, seq_len, hdim = out.shape
                prompt_embeds = out.permute(0, 2, 1, 3).reshape(B, seq_len, num_ch * hdim)
                # Text position ids for Flux2 RoPE: (B, L, 4)
                t = torch.arange(1, device=prompt_embeds.device, dtype=prompt_embeds.dtype)
                h = torch.arange(1, device=prompt_embeds.device, dtype=prompt_embeds.dtype)
                w = torch.arange(1, device=prompt_embeds.device, dtype=prompt_embeds.dtype)
                l = torch.arange(seq_len, device=prompt_embeds.device, dtype=prompt_embeds.dtype)
                text_ids = torch.cartesian_prod(t, h, w, l).unsqueeze(0).expand(B, -1, -1)
                return {
                    "embedding": prompt_embeds,
                    "encoder_hidden_states": prompt_embeds,
                    "txt_ids": text_ids,
                    "output": prompt_embeds,
                }

        emb = torch.zeros(1, self.max_length, self.embedding_dim)
        return {
            "embedding": emb,
            "encoder_hidden_states": emb,
            "output": emb,
        }
