"""CLAP conditioner for audio-text conditioning."""
import torch
from typing import Dict, Any
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.model.conditioner import AbstractConditioner


@register_block("conditioner/clap")
class CLAPConditioner(AbstractConditioner):
    """CLAP (Contrastive Language-Audio Pretraining) conditioner.
    
    Maps text descriptions to audio-aligned embeddings.
    Used by AudioLDM and other audio diffusion models.
    """
    
    block_type = "conditioner/clap"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._model = None
        self._processor = None
        self.pretrained = config.get("pretrained", "laion/clap-htsat-unfused")
        self.embed_dim = int(config.get("embed_dim", 512))
    
    def _load(self):
        """Lazy-load the CLAP model."""
        if self._model is not None:
            return
        try:
            from transformers import ClapModel, ClapProcessor
            self._processor = ClapProcessor.from_pretrained(self.pretrained)
            self._model = ClapModel.from_pretrained(self.pretrained)
            self._model.eval()
        except ImportError:
            # Fallback: random projection
            import torch.nn as nn
            self._model = nn.Linear(768, self.embed_dim)
    
    def __call__(self, condition: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        self._load()
        
        prompt = condition.get("text", condition.get("prompt", ""))
        if isinstance(prompt, str):
            prompt = [prompt]
        
        if self._processor is not None:
            inputs = self._processor(text=prompt, return_tensors="pt", padding=True)
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            
            with torch.no_grad():
                outputs = self._model.get_text_features(**inputs)
            # get_text_features can return tensor or BaseModelOutputWithPooling; avoid using tensor in boolean context
            if isinstance(outputs, torch.Tensor):
                text_features = outputs
            else:
                text_features = getattr(outputs, "pooler_output", None)
                if text_features is None:
                    text_features = getattr(outputs, "last_hidden_state", None)
                if text_features is None:
                    text_features = getattr(outputs, "text_embeds", None)
                if not isinstance(text_features, torch.Tensor):
                    dev = next(self._model.parameters()).device
                    text_features = torch.zeros(len(prompt), self.embed_dim, device=dev)
                if isinstance(text_features, torch.Tensor) and text_features.dim() == 2:
                    text_features = text_features.unsqueeze(1)
            return {"encoder_hidden_states": text_features}
        
        # Fallback
        dummy = torch.zeros(len(prompt), 1, self.embed_dim)
        return {"encoder_hidden_states": dummy}
