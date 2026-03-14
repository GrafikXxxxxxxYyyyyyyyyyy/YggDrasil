"""FLUX prompt encoding: CLIP (pooled) + T5 (sequence embeddings)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.diffusion import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractConjector


class FluxPromptEncoderNode(AbstractConjector):
    """Encodes prompts through CLIP (pooled) and T5 (sequence) for FLUX.

    CLIP produces pooled_prompt_embeds [B, 768].
    T5 produces prompt_embeds [B, seq_len, 4096] with up to 512 tokens.
    text_ids are zeros of shape [seq_len, 3] for RoPE positional encoding.

    Unlike SD/SDXL, FLUX does not use negative prompts (no CFG).
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        tokenizer: Any = None,
        tokenizer_2: Any = None,
        text_encoder: Any = None,
        text_encoder_2: Any = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)
        self._tokenizer = tokenizer
        self._tokenizer_2 = tokenizer_2
        self._text_encoder = text_encoder
        self._text_encoder_2 = text_encoder_2

    @property
    def block_type(self) -> str:
        return "flux/prompt_encoder"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_PROMPT, PortDirection.IN, PortType.TEXT, optional=True),
            Port(C.PORT_PROMPT_2, PortDirection.IN, PortType.TEXT, optional=True),
            Port(C.PORT_PROMPT_EMBEDS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_POOLED_PROMPT_EMBEDS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_TXT_IDS, PortDirection.OUT, PortType.TENSOR),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = inputs.get(C.PORT_PROMPT, "")
        prompt_2 = inputs.get(C.PORT_PROMPT_2) or prompt
        max_seq_len = self._config.get("max_sequence_length", 512)

        pooled_prompt_embeds = self._encode_clip(prompt)

        prompt_embeds, text_ids = self._encode_t5(prompt_2, max_seq_len)

        return {
            C.PORT_PROMPT_EMBEDS: prompt_embeds,
            C.PORT_POOLED_PROMPT_EMBEDS: pooled_prompt_embeds,
            C.PORT_TXT_IDS: text_ids,
        }

    def _encode_clip(self, text: str) -> Any:
        """Encode with CLIP to get pooled embeddings."""
        text_inputs = self._tokenizer(
            text,
            padding="max_length",
            max_length=self._tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self._text_encoder.device)
        output = self._text_encoder(input_ids, output_hidden_states=False)
        return output.pooler_output

    def _encode_t5(self, text: str, max_sequence_length: int) -> Any:
        """Encode with T5 to get sequence embeddings and text_ids."""
        import torch

        text_inputs = self._tokenizer_2(
            text,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self._text_encoder_2.device)
        output = self._text_encoder_2(input_ids)
        prompt_embeds = output[0]

        seq_len = prompt_embeds.shape[1]
        text_ids = torch.zeros(seq_len, 3, device=prompt_embeds.device)

        return prompt_embeds, text_ids

    def to(self, device: Any) -> "FluxPromptEncoderNode":
        for enc in (self._text_encoder, self._text_encoder_2):
            if enc is not None and hasattr(enc, "to"):
                enc.to(device)
        return self
