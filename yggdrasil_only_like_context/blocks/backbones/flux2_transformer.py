"""FLUX.2 Transformer backbone — Flux2Transformer2DModel wrapper.

Based on diffusers v0.36.0 Flux2Transformer2DModel:
- Dual-stream (8 layers) + Single-stream (48 layers) DiT
- SwiGLU activation, QK-norm (RMSNorm)
- Parallel attention + FF in single-stream blocks
- Guidance-distilled: guidance_scale is embedded, NOT traditional CFG
- Latent channels: 128 (32ch patchified 2x2)
- Text encoder: Mistral3 (joint_attention_dim=15360)
- VAE: AutoencoderKLFlux2

Wraps diffusers.Flux2Transformer2DModel.
Expects x (B, 32, H, W); patchifies to (B, 128, h, w), packs to (B, h*w, 128), then unpacks+unpatchifies output.
Falls back to MMDiT stub if diffusers unavailable.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.block.port import InputPort, OutputPort, TensorSpec, Port
from ...core.model.backbone import AbstractBackbone


@register_block("backbone/flux2_transformer")
class Flux2TransformerBackbone(AbstractBackbone):
    """FLUX.2 Transformer backbone (all variants).

    Architecture (from diffusers v0.36.0):
        - Dual-stream blocks: 8 (Flux2TransformerBlock)
        - Single-stream blocks: 48 (Flux2SingleTransformerBlock)
        - Hidden dim: 6144 (48 heads * 128 dim_head)
        - in_channels: 128 (after 2x2 patchification of 32ch latents)
        - joint_attention_dim: 15360 (Mistral3 3-layer hidden)
        - Guidance-distilled: guidance as embedding input, not CFG

    Variants: dev, schnell, fill, canny, depth, redux, kontext
    """

    block_type = "backbone/flux2_transformer"

    def __init__(self, config: DictConfig | dict):
        super().__init__(config)
        self._model = None
        self.pretrained = self.config.get("pretrained", "black-forest-labs/FLUX.2-dev")
        self.variant = self.config.get("variant", "dev")
        self._build_model()

    @classmethod
    def declare_io(cls) -> Dict[str, Port]:
        ports = dict(AbstractBackbone.declare_io())
        ports["guidance"] = InputPort(
            "guidance", data_type="tensor", optional=True,
            description="Guidance scale (embedded, for guidance-distilled model)",
        )
        ports["img_ids"] = InputPort(
            "img_ids", data_type="tensor", optional=True,
            description="Image position IDs for RoPE",
        )
        ports["txt_ids"] = InputPort(
            "txt_ids", data_type="tensor", optional=True,
            description="Text position IDs for RoPE",
        )
        ports["image_latents"] = InputPort(
            "image_latents", data_type="tensor", optional=True,
            description="Reference image latents (for kontext/redux/fill variants)",
        )
        return ports

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """(B, 32, H, W) -> (B, 128, H//2, W//2)."""
        B, C, H, W = latents.shape
        latents = latents.view(B, C, H // 2, 2, W // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(B, C * 4, H // 2, W // 2)

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """(B, 128, h, w) -> (B, 32, h*2, w*2)."""
        B, C4, h, w = latents.shape
        C = C4 // 4
        latents = latents.reshape(B, C, 2, 2, h, w)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(B, C, h * 2, w * 2)

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """(B, C, h, w) -> (B, h*w, C)."""
        B, C, h, w = latents.shape
        return latents.reshape(B, C, h * w).permute(0, 2, 1)

    @staticmethod
    def _unpack_latents(packed: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """(B, h*w, C) -> (B, C, h, w)."""
        B, seq, C = packed.shape
        return packed.permute(0, 2, 1).reshape(B, C, h, w)

    @staticmethod
    def _prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
        """(B, C, h, w) -> (B, h*w, 4) position ids for RoPE (t, h, w, l)."""
        B, _, height, width = latents.shape
        t = torch.arange(1, device=latents.device, dtype=latents.dtype)
        h = torch.arange(height, device=latents.device, dtype=latents.dtype)
        w = torch.arange(width, device=latents.device, dtype=latents.dtype)
        l = torch.arange(1, device=latents.device, dtype=latents.dtype)
        latent_ids = torch.cartesian_prod(t, h, w, l)
        return latent_ids.unsqueeze(0).expand(B, -1, -1)

    @staticmethod
    def _prepare_text_ids_from_embeds(encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """(B, L, D) -> (B, L, 4) text position ids for RoPE."""
        B, L, _ = encoder_hidden_states.shape
        device = encoder_hidden_states.device
        dtype = encoder_hidden_states.dtype
        t = torch.arange(1, device=device, dtype=dtype)
        h = torch.arange(1, device=device, dtype=dtype)
        w = torch.arange(1, device=device, dtype=dtype)
        l = torch.arange(L, device=device, dtype=dtype)
        text_ids = torch.cartesian_prod(t, h, w, l)
        return text_ids.unsqueeze(0).expand(B, -1, -1)

    @staticmethod
    def _materialize_meta_tensors(module: nn.Module, device: str = "cpu"):
        """Replace meta-device parameters/buffers with materialized (empty) tensors so .to(device) works."""
        for mod in module.modules():
            for n, p in list(mod.named_parameters(recurse=False)):
                if p.is_meta:
                    setattr(mod, n, nn.Parameter(p.to_empty(device=device), requires_grad=p.requires_grad))
            for n, b in list(mod.named_buffers(recurse=False)):
                if b.is_meta:
                    setattr(mod, n, b.to_empty(device=device))

    def _build_model(self):
        try:
            from diffusers import Flux2Transformer2DModel
            load_kwargs = {
                "subfolder": "transformer",
                "torch_dtype": torch.bfloat16 if self.config.get("bf16", True) else torch.float32,
            }
            if self.config.get("token") is not None:
                load_kwargs["token"] = self.config.get("token")
            self._model = Flux2Transformer2DModel.from_pretrained(
                self.pretrained,
                **load_kwargs,
            )
            self._model.requires_grad_(False)
            # Klein: some weights (e.g. time_guidance_embed) are not in checkpoint → meta tensors.
            # Materialize them so .to(device) later does not fail.
            self._materialize_meta_tensors(self._model, device="cpu")
        except Exception:
            # Fallback: MMDiT stub
            try:
                from .mmdit import MMDiTBackbone
                fallback_config = {
                    "type": "backbone/mmdit",
                    "hidden_dim": int(self.config.get("hidden_dim", 3072)),
                    "num_layers": int(self.config.get("num_layers", 8)),
                    "num_heads": int(self.config.get("num_heads", 24)),
                    "in_channels": int(self.config.get("in_channels", 128)),
                    "patch_size": 1,
                }
                self._model = MMDiTBackbone(DictConfig(fallback_config))
            except Exception:
                # Minimal stub
                in_ch = int(self.config.get("in_channels", 128))
                hdim = int(self.config.get("hidden_dim", 3072))
                self._model = nn.Sequential(
                    nn.Linear(in_ch, hdim),
                    nn.SiLU(),
                    nn.Linear(hdim, in_ch),
                )

    def to(self, device=None, dtype=None):
        """Move to device; materialize any remaining meta tensors first so .to() does not fail (Klein)."""
        if self._model is not None and device is not None:
            try:
                self._materialize_meta_tensors(self._model, device=str(device))
            except Exception:
                pass
        return super().to(device=device, dtype=dtype) if dtype is not None else super().to(device)

    def _forward_impl(self, x, timestep, condition=None, position_embedding=None, **kwargs):
        if condition is not None and isinstance(condition, dict):
            encoder_hidden_states = condition.get("encoder_hidden_states")
            txt_ids = condition.get("txt_ids")
        elif condition is not None and hasattr(condition, "shape"):
            encoder_hidden_states = condition
            txt_ids = None
        else:
            encoder_hidden_states = None
            txt_ids = None
        if txt_ids is None and encoder_hidden_states is not None:
            txt_ids = self._prepare_text_ids_from_embeds(encoder_hidden_states)
        guidance = kwargs.get("guidance")
        img_ids = kwargs.get("img_ids")
        if isinstance(condition, dict) and img_ids is None:
            img_ids = condition.get("img_ids")

        # Flux2Transformer2DModel expects (B, seq, 128); we get (B, 32, H, W) -> patchify + pack
        unpack_shape = None
        if hasattr(self._model, "config") and getattr(self._model.config, "in_channels", 0) == 128:
            if x.dim() == 4 and x.shape[1] == 32:
                B, C, H, W = x.shape
                x = self._patchify_latents(x)
                _, _, h, w = x.shape
                if img_ids is None:
                    img_ids = self._prepare_latent_ids(x)
                x = self._pack_latents(x)
                unpack_shape = (h, w)

        if hasattr(self._model, 'forward') and hasattr(self._model, 'config'):
            try:
                model_dtype = next(self._model.parameters()).dtype
                model_device = next(self._model.parameters()).device
                x = x.to(device=model_device, dtype=model_dtype)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(device=model_device, dtype=model_dtype)
                cfg = getattr(self._model, 'config', None)
                if guidance is None and cfg is not None and getattr(cfg, 'guidance_embeds', True) is False:
                    ch = getattr(cfg, 'timestep_guidance_channels', 256)
                    guidance = torch.zeros(x.shape[0], ch, device=model_device, dtype=model_dtype)
                fwd_kwargs = {
                    "hidden_states": x,
                    "timestep": timestep.to(device=model_device),
                    "encoder_hidden_states": encoder_hidden_states,
                    "return_dict": False,
                }
                if guidance is not None:
                    fwd_kwargs["guidance"] = guidance
                if img_ids is not None:
                    fwd_kwargs["img_ids"] = img_ids.to(device=model_device)
                if txt_ids is not None:
                    fwd_kwargs["txt_ids"] = txt_ids.to(device=model_device)
                out = self._model(**fwd_kwargs)[0]
                if unpack_shape is not None:
                    h, w = unpack_shape
                    out = self._unpack_latents(out, h, w)
                    out = self._unpatchify_latents(out)
                return out.float()
            except (TypeError, RuntimeError):
                if unpack_shape is not None:
                    raise
                pass

        # Fallback for stubs
        if hasattr(self._model, '_forward_impl'):
            return self._model._forward_impl(x, timestep, condition, **kwargs)
        if isinstance(self._model, nn.Sequential):
            return self._model(x)
        return x
