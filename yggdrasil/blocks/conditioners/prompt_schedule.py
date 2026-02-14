# yggdrasil/blocks/conditioners/prompt_schedule.py
"""Prompt Travel: кондиционер с расписанием промптов по кадрам/времени.

Подключается вместо обычного conditioner в любом пайплайне (video/image).
Вход: prompt_schedule = [(frame_start, frame_end, "prompt text"), ...] или
      list of {"frames": (start, end), "prompt": str}.
Выход: embedding (при per_frame_output=True — [B, num_frames, seq_len, dim] для
      по-кадрового контроля в backbone; иначе усреднённый по текущему шагу).
"""
from __future__ import annotations

import torch
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional, Union

from ...core.block.registry import register_block
from ...core.block.port import InputPort, OutputPort
from ..conditioners.clip_text import CLIPTextConditioner


@register_block("conditioner/prompt_schedule")
class PromptScheduleConditioner(CLIPTextConditioner):
    """Кондиционер с расписанием промптов по кадрам (Prompt Travel).

    Конфиг: как conditioner/clip_text + per_frame_output: bool.
    Вход raw_condition может быть:
      - {"text": "one prompt"} — один промпт на всё (как обычно);
      - {"prompt_schedule": [(0, 8, "prompt A"), (8, 16, "prompt B")], "num_frames": 16}
    Выход при per_frame_output=True: [B, num_frames, seq_len, dim] для backbone с поддержкой
    по-кадрового условия; иначе [B, seq_len, dim] (усреднение по сегменту для текущего шага).
    """

    block_type = "conditioner/prompt_schedule"

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.per_frame_output = config.get("per_frame_output", False)

    @classmethod
    def declare_io(cls):
        return {
            "raw_condition": InputPort(
                "raw_condition",
                data_type="dict",
                description="Dict: 'text' | 'prompt_schedule' (list of (start, end, prompt)), 'num_frames'",
            ),
            "embedding": OutputPort("embedding", description="Text embedding(s) for conditioning"),
        }

    def process(self, **port_inputs) -> Dict[str, Any]:
        raw = port_inputs.get("raw_condition", {})
        if isinstance(raw, str):
            raw = {"text": raw}
        prompt_schedule = raw.get("prompt_schedule")
        num_frames = raw.get("num_frames", 16)
        if not prompt_schedule:
            return super().process(**port_inputs)

        embeddings_per_segment = []
        for item in prompt_schedule:
            if isinstance(item, (list, tuple)):
                start, end, text = item[0], item[1], item[2]
            else:
                start, end = item["frames"]
                text = item["prompt"]
            emb = super().process(raw_condition={"text": text}, **port_inputs)
            e = emb["embedding"]
            embeddings_per_segment.append((start, end, e))
        if not embeddings_per_segment:
            return super().process(**port_inputs)

        device = embeddings_per_segment[0][2].device
        dtype = embeddings_per_segment[0][2].dtype
        B, seq_len, dim = (
            embeddings_per_segment[0][2].shape[0],
            embeddings_per_segment[0][2].shape[1],
            embeddings_per_segment[0][2].shape[2],
        )
        out = torch.zeros(B, num_frames, seq_len, dim, device=device, dtype=dtype)
        for start, end, e in embeddings_per_segment:
            end = min(end, num_frames)
            n = end - start
            if n <= 0:
                continue
            e_rep = e.expand(n, -1, -1) if e.shape[0] == 1 else e.repeat(n, 1, 1)
            out[:, start:end] = e_rep.unsqueeze(0).expand(B, n, seq_len, dim).to(device=device, dtype=dtype)
        if self.per_frame_output:
            return {"embedding": out}
        return {"embedding": out.mean(dim=1)}
