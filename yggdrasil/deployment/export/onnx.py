"""ONNX export for YggDrasil model backbones."""
from __future__ import annotations

import torch
from pathlib import Path
from typing import Optional, Tuple


def export_backbone_to_onnx(
    model,
    output_path: str | Path,
    input_shape: Tuple[int, ...] = (1, 4, 64, 64),
    opset_version: int = 17,
    dynamic_axes: bool = True,
) -> Path:
    """Export the backbone of a ModularDiffusionModel to ONNX.
    
    Args:
        model: ModularDiffusionModel
        output_path: Output .onnx file path
        input_shape: Example input tensor shape
        opset_version: ONNX opset version
        dynamic_axes: Whether to use dynamic axes for batch/spatial dims
        
    Returns:
        Path to the exported ONNX model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    backbone = model._slot_children.get("backbone")
    if backbone is None:
        raise ValueError("Model has no backbone to export")
    
    backbone.eval()
    device = next(backbone.parameters()).device
    
    # Create dummy inputs
    dummy_x = torch.randn(input_shape, device=device)
    dummy_t = torch.tensor([500], device=device)
    
    # Build dynamic axes
    axes = None
    if dynamic_axes:
        axes = {
            "latents": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }
    
    torch.onnx.export(
        backbone,
        (dummy_x, dummy_t),
        str(output_path),
        opset_version=opset_version,
        input_names=["latents", "timestep"],
        output_names=["output"],
        dynamic_axes=axes,
    )
    
    print(f"Backbone exported to ONNX: {output_path}")
    return output_path


def export_vae_to_onnx(
    model,
    output_dir: str | Path,
    input_shape: Tuple[int, ...] = (1, 3, 512, 512),
    opset_version: int = 17,
) -> dict:
    """Export the VAE encoder and decoder separately.
    
    Returns:
        Dict with 'encoder' and 'decoder' paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    codec = model._slot_children.get("codec")
    if codec is None:
        raise ValueError("Model has no codec to export")
    
    paths = {}
    
    # Export encoder
    if hasattr(codec, "vae"):
        vae = codec.vae
        vae.eval()
        device = next(vae.parameters()).device
        
        dummy_input = torch.randn(input_shape, device=device, dtype=vae.dtype)
        encoder_path = output_dir / "vae_encoder.onnx"
        
        torch.onnx.export(
            vae.encoder,
            dummy_input,
            str(encoder_path),
            opset_version=opset_version,
            input_names=["input"],
            output_names=["latent"],
        )
        paths["encoder"] = encoder_path
        
        # Export decoder
        latent_shape = (input_shape[0], codec.latent_channels,
                        input_shape[2] // codec.spatial_scale_factor,
                        input_shape[3] // codec.spatial_scale_factor)
        dummy_latent = torch.randn(latent_shape, device=device, dtype=vae.dtype)
        decoder_path = output_dir / "vae_decoder.onnx"
        
        torch.onnx.export(
            vae.decoder,
            dummy_latent,
            str(decoder_path),
            opset_version=opset_version,
            input_names=["latent"],
            output_names=["output"],
        )
        paths["decoder"] = decoder_path
    
    print(f"VAE exported to ONNX: {output_dir}")
    return paths
