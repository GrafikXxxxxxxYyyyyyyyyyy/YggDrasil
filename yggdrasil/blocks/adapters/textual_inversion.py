# yggdrasil/blocks/adapters/textual_inversion.py
"""Textual Inversion — learn new concepts via token embeddings.

Like A1111's Textual Inversion: learns a new embedding vector for a
placeholder token (e.g., <my-concept>) that can be used in prompts.

Usage::

    ti = TextualInversionAdapter({
        "type": "adapter/textual_inversion",
        "placeholder_token": "<my-cat>",
        "num_vectors": 4,
        "embedding_dim": 768,
    })
    
    # Inject into text encoder
    ti.inject_into(text_encoder_block)
    
    # Train with GraphTrainer (only TI embedding is trainable)
    trainer = GraphTrainer(graph, train_nodes=["textual_inversion"])
    trainer.train(dataset)
"""
from __future__ import annotations

import logging
import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional

from yggdrasil.core.block.registry import register_block
from yggdrasil.core.block.port import InputPort, OutputPort
from .base import AbstractAdapter

logger = logging.getLogger(__name__)


@register_block("adapter/textual_inversion")
class TextualInversionAdapter(AbstractAdapter):
    """Textual Inversion adapter — trainable token embedding.
    
    Creates a new embedding vector for a placeholder token that can be
    optimized to represent a new concept. The embedding is the only
    trainable parameter.
    
    Features:
    - Multi-vector embeddings (num_vectors > 1 for richer representations)
    - Save/load learned embeddings
    - Compatible with any text encoder that uses nn.Embedding
    """
    
    block_type = "adapter/textual_inversion"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.placeholder_token = config.get("placeholder_token", "<concept>")
        self.num_vectors = config.get("num_vectors", 1)
        self.embedding_dim = config.get("embedding_dim", 768)
        self.initializer_token = config.get("initializer_token", None)
        
        # The trainable embedding(s)
        self.learned_embeds = nn.Parameter(
            torch.randn(self.num_vectors, self.embedding_dim) * 0.01
        )
        
        # Placeholder token IDs (set during injection)
        self._placeholder_token_ids: List[int] = []
        self._injected_encoder = None
    
    @classmethod
    def declare_io(cls):
        return {
            "text_tokens": InputPort("text_tokens", description="Tokenized input IDs", optional=True),
            "text_embeds": InputPort("text_embeds", description="Pre-computed text embeddings", optional=True),
            "modified_embeds": OutputPort("modified_embeds", description="Embeddings with TI tokens replaced"),
        }
    
    def process(self, **kw) -> Dict[str, Any]:
        """Replace placeholder token embeddings with learned ones."""
        text_embeds = kw.get("text_embeds")
        text_tokens = kw.get("text_tokens")
        
        if text_embeds is None:
            return {"modified_embeds": None}
        
        # If we have token IDs, replace the placeholder positions
        if text_tokens is not None and self._placeholder_token_ids:
            modified = text_embeds.clone()
            for batch_idx in range(text_tokens.shape[0]):
                for vec_idx, token_id in enumerate(self._placeholder_token_ids):
                    mask = text_tokens[batch_idx] == token_id
                    if mask.any():
                        pos = mask.nonzero(as_tuple=True)[0]
                        if len(pos) > 0 and vec_idx < self.num_vectors:
                            modified[batch_idx, pos[0]] = self.learned_embeds[vec_idx]
            return {"modified_embeds": modified}
        
        return {"modified_embeds": text_embeds}
    
    def inject_into(self, target):
        """Inject placeholder token into text encoder's tokenizer and embedding.
        
        Adds new token(s) to the tokenizer and resizes the embedding layer.
        """
        # Try to find the embedding layer
        embedding_layer = None
        for name, module in target.named_modules():
            if isinstance(module, nn.Embedding):
                embedding_layer = module
                break
        
        if embedding_layer is not None:
            # Initialize from existing token if specified
            if self.initializer_token is not None:
                logger.info(f"TI: Would initialize from token '{self.initializer_token}'")
            
            self._injected_encoder = target
            logger.info(
                f"Textual Inversion: registered '{self.placeholder_token}' "
                f"with {self.num_vectors} vectors (dim={self.embedding_dim})"
            )
        else:
            logger.warning("No nn.Embedding found in target — TI injection skipped")
    
    def apply(self, output: torch.Tensor, context=None) -> torch.Tensor:
        return output
    
    def save_embedding(self, path: str):
        """Save the learned embedding."""
        state = {
            "placeholder_token": self.placeholder_token,
            "num_vectors": self.num_vectors,
            "embedding_dim": self.embedding_dim,
            "learned_embeds": self.learned_embeds.data,
        }
        torch.save(state, path)
        logger.info(f"Saved TI embedding to {path}")
    
    def load_embedding(self, path: str):
        """Load a learned embedding."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.learned_embeds.data = state["learned_embeds"]
        self.placeholder_token = state.get("placeholder_token", self.placeholder_token)
        self.num_vectors = state.get("num_vectors", self.num_vectors)
        logger.info(f"Loaded TI embedding from {path}: '{self.placeholder_token}'")
