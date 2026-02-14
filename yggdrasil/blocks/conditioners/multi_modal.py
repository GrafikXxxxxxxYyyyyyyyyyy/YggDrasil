"""Multi-modal conditioner that combines multiple conditioners."""
import torch
from typing import Dict, Any, List
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.block.slot import Slot
from ...core.model.conditioner import AbstractConditioner


@register_block("conditioner/multi_modal")
class MultiModalConditioner(AbstractConditioner):
    """Combines multiple conditioners into one.
    
    Merges outputs from text, image, audio, and other conditioners
    into a single condition dict.
    """
    
    block_type = "conditioner/multi_modal"
    
    def _define_slots(self):
        return {
            "conditioners": Slot(
                name="conditioners",
                accepts=AbstractConditioner,
                multiple=True,
                optional=True,
            )
        }
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.fusion = config.get("fusion", "concat")  # concat, add, cross_attention
    
    def __call__(self, condition: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        all_embeds = {}
        
        conditioners = self._slot_children.get("conditioners", [])
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
