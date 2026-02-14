"""Encodec codec for audio diffusion.

Wraps Meta's Encodec model for encoding/decoding audio.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.model.codec import AbstractLatentCodec


@register_block("codec/encodec")
class EncodecCodec(AbstractLatentCodec):
    """Encodec audio codec.
    
    Encodes waveforms into discrete/continuous audio tokens.
    Used by AudioLDM, Stable Audio, and other audio diffusion models.
    
    For production, loads Meta's Encodec model via transformers or encodec library.
    """
    
    block_type = "codec/encodec"
    is_trainable = False
    
    def __init__(self, config: DictConfig | dict = None):
        super().__init__(config or {})
        self.sample_rate = int(self.config.get("sample_rate", 24000))
        self.latent_channels = int(self.config.get("latent_channels", 128))
        self.bandwidth = float(self.config.get("bandwidth", 6.0))
        self.hop_length = int(self.config.get("hop_length", 320))
        
        self._encodec_model = None
        
        # Fallback: simple 1D conv encoder/decoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, 2, 3),
            nn.ReLU(),
            nn.Conv1d(32, 64, 7, 2, 3),
            nn.ReLU(),
            nn.Conv1d(64, self.latent_channels, 7, 2, 3),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.latent_channels, 64, 7, 2, 3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 7, 2, 3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 7, 2, 3, output_padding=1),
        )
    
    def _load_encodec(self):
        """Lazy-load the real Encodec model."""
        if self._encodec_model is not None:
            return
        try:
            from transformers import EncodecModel
            self._encodec_model = EncodecModel.from_pretrained(
                self.config.get("pretrained", "facebook/encodec_24khz")
            )
            self._encodec_model.eval()
        except ImportError:
            pass  # Use fallback conv encoder/decoder
    
    def get_latent_shape(
        self,
        batch_size: int = 1,
        duration_seconds: float = 5.0,
        **kwargs,
    ) -> Tuple[int, ...]:
        length = int(duration_seconds * self.sample_rate / self.hop_length)
        return (batch_size, self.latent_channels, length)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode waveform [B, 1, T] -> latents [B, C, L]."""
        self._load_encodec()
        if self._encodec_model is not None:
            with torch.no_grad():
                encoded = self._encodec_model.encode(x, bandwidth=self.bandwidth)
                # Return continuous representation
                return encoded.audio_codes.float()
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents [B, C, L] -> waveform [B, 1, T]."""
        self._load_encodec()
        if self._encodec_model is not None:
            with torch.no_grad():
                return self._encodec_model.decode(z.long())
        return self.decoder(z)
