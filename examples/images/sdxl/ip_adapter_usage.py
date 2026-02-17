"""IP-Adapter usage examples — mapping Diffusers docs to YggDrasil.

Each section corresponds to a feature from:
https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter

Quick reference:
    pipe(prompt="...", ip_image=img)                    # Basic
    pipe(..., ip_image_embeds=embeds)                   # Precomputed
    pipe(..., ip_adapter_scale=0.8)                     # Scale
    pipe(..., ip_adapter_scale={"down": {...}})         # InstantStyle per-layer
    pipe(..., ip_adapter_masks=masks)                   # Masking
    pipe(..., ip_image_plus=img)                        # IP-Adapter Plus
    pipe(..., ip_face_image=face_img)                  # IP-Adapter FaceID
"""
from __future__ import annotations

# =============================================================================
# 1. BASIC: Load IP-Adapter, set scale, generate with ip_adapter_image
# Diffusers: load_ip_adapter(), set_ip_adapter_scale(), ip_adapter_image=
# =============================================================================

def example_1_basic_txt2img():
    """Diffusers: pipeline.load_ip_adapter() + set_ip_adapter_scale() + ip_adapter_image=image"""
    from yggdrasil import InferencePipeline

    pipe = InferencePipeline.from_template("sdxl_txt2img", device="cuda")
    pipe.graph.to("cuda")

    # Add IP-Adapter to graph (equivalent to load_ip_adapter)
    from yggdrasil.core.graph.adapters import add_ip_adapter_to_graph
    add_ip_adapter_to_graph(pipe.graph, ip_adapter_scale=0.8)

    # Load weights into adapter block
    adapter = pipe.graph.nodes["ip_adapter"]
    adapter.load_weights("path/to/ip-adapter_sdxl.bin")  # or HF: h94/IP-Adapter sdxl_models

    # Generate (ip_adapter_image = single image)
    out = pipe(
        prompt="a polar bear sitting in a chair drinking a milkshake",
        ip_image="https://example.com/reference.png",
        negative_prompt="deformed, ugly, wrong proportion, low res",
        num_steps=50,
    )
    return out.images[0]


# =============================================================================
# 2. IMAGE-TO-IMAGE + IP-Adapter
# Diffusers: AutoPipelineForImage2Image + ip_adapter_image=
# =============================================================================

def example_2_img2img_ip():
    """Diffusers: img2img pipeline + ip_adapter_image"""
    from yggdrasil import InferencePipeline

    pipe = InferencePipeline.from_template("sdxl_img2img", device="cuda")
    from yggdrasil.core.graph.adapters import add_ip_adapter_to_graph
    add_ip_adapter_to_graph(pipe.graph, ip_adapter_scale=0.8)

    out = pipe(
        prompt="best quality, high quality",
        init_image="path/to/input.png",  # base image
        ip_image="path/to/style_reference.png",  # IP conditioning
        strength=0.5,
    )
    return out.images[0]


# =============================================================================
# 3. IMAGE EMBEDDINGS: prepare_ip_adapter_image_embeds, ip_adapter_image_embeds
# Diffusers: prepare_ip_adapter_image_embeds() + ip_adapter_image_embeds=
# =============================================================================

def example_3_precomputed_embeds():
    """Diffusers: prepare_ip_adapter_image_embeds + save/load + ip_adapter_image_embeds"""
    import torch
    from yggdrasil import InferencePipeline

    pipe = InferencePipeline.from_template("sdxl_txt2img", device="cuda")
    from yggdrasil.core.graph.adapters import add_ip_adapter_to_graph
    add_ip_adapter_to_graph(pipe.graph, ip_adapter_scale=0.8)

    # Precompute embeddings (equivalent to pipeline.prepare_ip_adapter_image_embeds)
    embeds = pipe.prepare_ip_adapter_embeds(
        ip_adapter_image="path/to/reference.png",
        ip_adapter_scale=0.8,
        device="cuda",
    )
    torch.save(embeds, "image_embeds.ipadpt")

    # Later: bypass encoder, use saved embeds
    embeds = torch.load("image_embeds.ipadpt", map_location="cuda")
    out = pipe(
        prompt="a polar bear sitting in a chair drinking a milkshake",
        ip_image_embeds=embeds,
        negative_prompt="deformed, ugly",
    )
    return out.images[0]


# =============================================================================
# 4. IP-Adapter PLUS (ViT-H, patch embeddings)
# Diffusers: load_ip_adapter(weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
# =============================================================================

def example_4_ip_adapter_plus():
    """Diffusers: IP-Adapter Plus with ViT-H patch embeddings"""
    from yggdrasil import InferencePipeline
    from yggdrasil.core.graph.adapters import add_ip_adapter_plus_to_graph

    pipe = InferencePipeline.from_template("sdxl_txt2img", device="cuda")
    add_ip_adapter_plus_to_graph(pipe.graph, ip_adapter_scale=0.8)

    # Load Plus weights
    pipe.graph.nodes["ip_adapter_plus"].load_weights(
        "path/to/ip-adapter-plus_sdxl_vit-h.safetensors"
    )

    out = pipe(
        prompt="a cat, masterpiece, best quality",
        ip_image_plus="path/to/style_reference.png",  # use ip_image_plus for Plus adapter
        negative_prompt="low quality",
    )
    return out.images[0]


# =============================================================================
# 5. IP-Adapter FaceID (InsightFace)
# Diffusers: load_ip_adapter("h94/IP-Adapter-FaceID", weight_name="ip-adapter-faceid_sdxl.bin")
# =============================================================================

def example_5_ip_adapter_faceid():
    """Diffusers: IP-Adapter FaceID with InsightFace embeddings"""
    from yggdrasil import InferencePipeline
    from yggdrasil.core.graph.adapters import add_ip_adapter_faceid_to_graph

    pipe = InferencePipeline.from_template("sdxl_txt2img", device="cuda")
    add_ip_adapter_faceid_to_graph(pipe.graph, ip_adapter_scale=0.6)

    pipe.graph.nodes["ip_adapter_faceid"].load_weights(
        "path/to/ip-adapter-faceid_sdxl.bin"
    )

    out = pipe(
        prompt="A photo of a girl",
        ip_face_image="path/to/face_crop.png",  # cropped face image
        negative_prompt="monochrome, lowres, bad anatomy",
    )
    return out.images[0]


# =============================================================================
# 6. FaceID with precomputed embeddings (ip_adapter_image_embeds)
# Diffusers: extract FaceID embeds, pass as ip_adapter_image_embeds
# =============================================================================

def example_6_faceid_precomputed_embeds():
    """Diffusers: InsightFace FaceAnalysis -> ip_adapter_image_embeds"""
    import torch
    from yggdrasil import InferencePipeline
    from yggdrasil.core.graph.adapters import add_ip_adapter_faceid_to_graph

    pipe = InferencePipeline.from_template("sd15_txt2img", device="cuda")
    add_ip_adapter_faceid_to_graph(pipe.graph, ip_adapter_scale=0.6)
    pipe.graph.nodes["ip_adapter_faceid"].load_weights("path/to/ip-adapter-faceid_sd15.bin")

    # Extract face embeddings with InsightFace (or use conditioner/faceid block)
    from insightface.app import FaceAnalysis
    import numpy as np
    from PIL import Image

    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    img = np.array(Image.open("face.png").convert("RGB"))
    faces = app.get(img)
    emb = torch.from_numpy(faces[0].normed_embedding).float().unsqueeze(0).unsqueeze(1)

    # Project through adapter to get image_prompt_embeds
    adapter = pipe.graph.nodes["ip_adapter_faceid"]
    proj_out = adapter.process(image_features=emb)
    id_embeds = proj_out["image_prompt_embeds"].to("cuda", dtype=torch.float16)

    out = pipe(
        prompt="A photo of a girl",
        ip_image_embeds=id_embeds,
        negative_prompt="monochrome, lowres",
    )
    return out.images[0]


# =============================================================================
# 7. MASKING: IPAdapterMaskProcessor + cross_attention_kwargs ip_adapter_masks
# Diffusers: IPAdapterMaskProcessor.preprocess() + cross_attention_kwargs={"ip_adapter_masks": masks}
# =============================================================================

def example_7_masking():
    """Diffusers: region-specific IP with binary masks"""
    import torch
    from yggdrasil import InferencePipeline
    from yggdrasil.core.graph.adapters import add_ip_adapter_to_graph
    from yggdrasil.core.block.builder import BlockBuilder

    pipe = InferencePipeline.from_template("sdxl_txt2img", device="cuda")
    add_ip_adapter_to_graph(pipe.graph, ip_adapter_scale=0.7)

    # Preprocess masks (equivalent to IPAdapterMaskProcessor.preprocess)
    mask_block = BlockBuilder.build({"type": "conditioner/ip_adapter_mask"})
    masks_out = mask_block.process(
        masks=["path/to/mask1.png", "path/to/mask2.png"],
        height=1024,
        width=1024,
    )
    ip_masks = masks_out["output"]

    out = pipe(
        prompt="2 girls",
        ip_image={"images": ["path/to/face1.png", "path/to/face2.png"]},
        ip_adapter_masks=ip_masks,
        ip_adapter_scale=[0.7, 0.7],
        negative_prompt="monochrome, lowres",
    )
    return out.images[0]


# =============================================================================
# 8. MULTIPLE IP-Adapters (style + face)
# Diffusers: weight_name=[...] + ip_adapter_image=[style_images, face_image]
# =============================================================================

def example_8_multiple_ip_adapters():
    """Diffusers: combine IP-Adapter Plus (style) + FaceID (face)"""
    from yggdrasil import InferencePipeline
    from yggdrasil.core.graph.adapters import (
        add_ip_adapter_to_graph,
        add_ip_adapter_plus_to_graph,
        add_ip_adapter_faceid_to_graph,
    )

    pipe = InferencePipeline.from_template("sdxl_txt2img", device="cuda")

    # Add base + Plus + FaceID -> merge block wires all
    add_ip_adapter_to_graph(pipe.graph, ip_adapter_scale=0.5)
    add_ip_adapter_plus_to_graph(pipe.graph, ip_adapter_scale=0.7)
    add_ip_adapter_faceid_to_graph(pipe.graph, ip_adapter_scale=0.3)

    # Load weights for each
    pipe.graph.nodes["ip_adapter"].load_weights("path/to/ip-adapter_sdxl.bin")
    pipe.graph.nodes["ip_adapter_plus"].load_weights("path/to/ip-adapter-plus_sdxl_vit-h.safetensors")
    pipe.graph.nodes["ip_adapter_faceid"].load_weights("path/to/ip-adapter-faceid_sdxl.bin")

    # Pass style images and face image (order matches adapters in merge)
    out = pipe(
        prompt="wonderwoman",
        ip_image=["path/to/style1.png", "path/to/style2.png"],
        ip_image_plus="path/to/style_folder/img.png",
        ip_face_image="path/to/face.png",
        ip_adapter_scale=[0.5, 0.7, 0.3],
        negative_prompt="monochrome, lowres",
    )
    return out.images[0]


# =============================================================================
# 9. Per-layer scale (InstantStyle)
# Diffusers: set_ip_adapter_scale({"down": {"block_2": [0,1]}, "up": {"block_0": [0,1,0]}})
# =============================================================================

def example_9_instantstyle_per_layer_scale():
    """Diffusers: InstantStyle — IP only in specific blocks"""
    from yggdrasil import InferencePipeline
    from yggdrasil.core.graph.adapters import add_ip_adapter_to_graph

    pipe = InferencePipeline.from_template("sdxl_txt2img", device="cuda")
    add_ip_adapter_to_graph(pipe.graph, ip_adapter_scale=0.8)

    # Per-layer scale (handled by pipeline when ip_adapter_scale is dict)
    scale = {
        "down": {"block_2": [0.0, 1.0]},
        "up": {"block_0": [0.0, 1.0, 0.0]},
    }

    out = pipe(
        prompt="a cat, masterpiece, best quality, high quality",
        ip_image="path/to/style.png",
        ip_adapter_scale=scale,
        negative_prompt="text, watermark, lowres",
        guidance_scale=5,
    )
    return out.images[0]


# =============================================================================
# 10. ControlNet + IP-Adapter
# Diffusers: StableDiffusionControlNetPipeline + load_ip_adapter
# =============================================================================

def example_10_controlnet_ip_adapter():
    """Diffusers: ControlNet (depth/canny) + IP-Adapter"""
    from yggdrasil import InferencePipeline
    from yggdrasil.core.graph.adapters import add_controlnet_to_graph, add_ip_adapter_to_graph

    pipe = InferencePipeline.from_template("sd15_controlnet_txt2img", device="cuda")
    add_controlnet_to_graph(
        pipe.graph,
        controlnet_pretrained="lllyasviel/control_v11f1p_sd15_depth",
        conditioning_scale=0.8,
    )
    add_ip_adapter_to_graph(pipe.graph, ip_adapter_scale=0.7)

    out = pipe(
        prompt="best quality, high quality",
        control_image="path/to/depth_map.png",
        ip_image="path/to/style_reference.png",
        negative_prompt="monochrome, lowres",
    )
    return out.images[0]


# =============================================================================
# 11. Per-image scales (multiple images)
# Diffusers: ip_adapter_image=[img1, img2] + set_ip_adapter_scale([0.7, 0.3])
# =============================================================================

def example_11_per_image_scales():
    """Diffusers: multiple images with different scales"""
    from yggdrasil import InferencePipeline
    from yggdrasil.core.graph.adapters import add_ip_adapter_to_graph

    pipe = InferencePipeline.from_template("sdxl_txt2img", device="cuda")
    add_ip_adapter_to_graph(pipe.graph)

    out = pipe(
        prompt="a portrait",
        ip_image={
            "images": ["path/to/style1.png", "path/to/style2.png"],
            "scales": [0.7, 0.3],
        },
        ip_adapter_scale=[0.7, 0.3],
        negative_prompt="low quality",
    )
    return out.images[0]


# =============================================================================
# 12. Unload / disable IP-Adapter
# Diffusers: pipeline calls without ip_adapter_image use base model
# =============================================================================

def example_12_disable_ip_adapter():
    """Diffusers: use pipeline without IP when ip_image not passed"""
    from yggdrasil import InferencePipeline
    from yggdrasil.core.graph.adapters import add_ip_adapter_to_graph

    pipe = InferencePipeline.from_template("sdxl_txt2img", device="cuda")
    add_ip_adapter_to_graph(pipe.graph, ip_adapter_scale=0.8)

    # With IP
    out_ip = pipe(prompt="a cat", ip_image="path/to/ref.png")

    # Without IP (pipeline auto-disables IP processors when ip_image/ip_image_embeds absent)
    out_clean = pipe(prompt="a cat")

    return out_ip.images[0], out_clean.images[0]


# =============================================================================
# Run one example (edit __name__ block to run)
# =============================================================================

if __name__ == "__main__":
    # Uncomment to run:
    # img = example_1_basic_txt2img()
    # img = example_3_precomputed_embeds()
    # img = example_9_instantstyle_per_layer_scale()
    print("See functions above. Uncomment in __main__ to run.")
