"""Multi-modal conditioner that combines multiple conditioners."""
import torch
from typing import Dict, Any, List
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.block.builder import BlockBuilder
from ...core.model.conditioner import AbstractConditioner


@register_block("conditioner/multi_modal")
class MultiModalConditioner(AbstractConditioner):
    """Combines multiple conditioners — built from config (no slots)."""

    block_type = "conditioner/multi_modal"

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.fusion = config.get("fusion", "concat")
        self._conditioners: List[Any] = []
        for c in config.get("conditioners", []) or []:
            if isinstance(c, dict) and (c.get("type") or c.get("block_type")):
                self._conditioners.append(BlockBuilder.build(c))
            elif hasattr(c, "block_type"):
                self._conditioners.append(c)

    def __call__(self, condition: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        all_embeds = {}
        conditioners = self._conditioners
        for cond in conditioners:
            embeds = cond(condition)
            all_embeds.update(embeds)
        
        # If multiple encoder_hidden_states, concatenate them
        hidden_states_list = []
        for key in list(all_embeds.keys()):
            if "hidden_states" in key:
                hidden_states_list.append(all_embeds.pop(key))
        
        if hidden_states_list:
            if self.fusion == "concat":
                all_embeds["encoder_hidden_states"] = torch.cat(
                    hidden_states_list, dim=1
                )
            elif self.fusion == "add":
                result = hidden_states_list[0]
                for hs in hidden_states_list[1:]:
                    if hs.shape == result.shape:
                        result = result + hs
                all_embeds["encoder_hidden_states"] = result
        
        return all_embeds
