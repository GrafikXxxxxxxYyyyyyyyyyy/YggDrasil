# IP-Adapter: маппинг возможностей Diffusers → YggDrasil

Референс: [Diffusers IP-Adapter docs](https://github.com/huggingface/diffusers/blob/main/docs/source/en/using-diffusers/ip_adapter.md)

## Текущая реализация

- **conditioner/clip_vision** — CLIP ViT-L encoder
- **adapter/ip_adapter** — projection (image_embed_dim → cross_attn), concat multi-image tokens
- **Pipeline**: `ip_image`, `ip_adapter_scale` (float/list), per-image scales
- **UNet**: Diffusers IPAdapterAttnProcessor

## Маппинг возможностей

| Diffusers | YggDrasil (Lego/граф) | Статус |
|-----------|----------------------|--------|
| **Основное** |
| load_ip_adapter | BlockBuilder + load_weights на adapter/ip_adapter | ✅ |
| set_ip_adapter_scale(float) | set_ip_adapter_scale_on_unet | ✅ |
| ip_adapter_image | ip_image → encoder → ip_adapter | ✅ |
| **Image embeddings** |
| prepare_ip_adapter_image_embeds | pipeline.prepare_ip_adapter_embeds() | ✅ |
| ip_adapter_image_embeds | graph input ip_image_embeds (bypass encoder) | ✅ |
| **Множественные IP-Adapters** |
| weight_name=[...] | adapter/ip_adapter_merge для concat | ✅ |
| set_ip_adapter_scale([0.7, 0.3]) | set_ip_adapter_scale_on_unet(List) | ✅ |
| ip_adapter_image=[style_imgs, face_img] | add_ip_adapter_plus + add_ip_adapter_faceid → merge | ✅ |
| **Варианты моделей** |
| IP-Adapter Plus (ViT-H) | conditioner/clip_vision output_mode=patches + adapter/ip_adapter_plus | ✅ |
| IP-Adapter FaceID | conditioner/faceid + adapter/ip_adapter_faceid | ✅ |
| **Per-layer scale (InstantStyle)** |
| scale={"down":{"block_2":[0,1]}, "up":{"block_0":[0,1,0]}} | set_ip_adapter_scale_on_unet(dict) | ✅ |
| **Masking** |
| IPAdapterMaskProcessor | block conditioner/ip_adapter_mask | ✅ |
| ip_adapter_masks в cross_attention | backbone input + cross_attention_kwargs passthrough | ✅ |
| **Pipeline-level** |
| I2I + IP | img2img template + ip_image | ✅ (template) |
| Inpainting + IP | inpaint template + mask_image + ip_image | 📋 |
| Video (AnimateDiff) + IP | video template | 📋 |
| LCM + IP | lcm template + ip_image | 📋 |
| ControlNet + IP | add_controlnet + add_ip_adapter | ✅ |
| unload_ip_adapter | _apply_ip_adapter_switch(False) | ✅ |

## Архитектура блоков

```
[raw images] → [encoder] → [ip_adapter] → image_prompt_embeds
                    ↑              ↑
              ip_image      ip_image_embeds (optional bypass)
```

**Вариант с precomputed embeds:**
```
ip_image_embeds (tensor) ──→ [ip_adapter] → image_prompt_embeds
```
Когда передан ip_image_embeds, encoder не вызывается.

**Multiple IP-Adapters:**
```
[encoder1] ──┐
[encoder2] ──┼→ [ip_adapter_multi] → combined embeds
[encoder3] ──┘
```
Или один encoder + несколько проекций (разные веса).

## Реализованные компоненты

1. **ip_image_embeds bypass** — вход в encoder/ip_adapter для готовых эмбеддингов ✅
2. **prepare_ip_adapter_embeds** — метод pipeline для предкодирования ✅
3. **Per-layer scale** — dict в set_ip_adapter_scale_on_unet ✅
4. **conditioner/ip_adapter_mask** — препроцессинг масок ✅
5. **adapter/ip_adapter_plus** — ViT-H + patch projection (conditioner/clip_vision output_mode=patches) ✅
6. **conditioner/faceid** + **adapter/ip_adapter_faceid** — FaceID (InsightFace) ✅
7. **adapter/ip_adapter_merge** — concat нескольких IP-Adapter выходов ✅
8. **ip_adapter_masks** — backbone input → cross_attention_kwargs passthrough ✅

## Usage examples (Diffusers → YggDrasil)

See `examples/images/sdxl/ip_adapter_usage.py` for runnable code.

| Diffusers | YggDrasil |
|-----------|-----------|
| `pipeline.load_ip_adapter(...)` | `add_ip_adapter_to_graph()` + `adapter.load_weights(path)` |
| `pipeline.set_ip_adapter_scale(0.8)` | `add_ip_adapter_to_graph(ip_adapter_scale=0.8)` or `pipe(..., ip_adapter_scale=0.8)` |
| `ip_adapter_image=image` | `pipe(..., ip_image=image)` |
| `ip_adapter_image_embeds=embeds` | `pipe(..., ip_image_embeds=embeds)` |
| `prepare_ip_adapter_image_embeds()` | `pipe.prepare_ip_adapter_embeds()` |
| IP-Adapter Plus | `add_ip_adapter_plus_to_graph()` + `pipe(..., ip_image_plus=...)` |
| IP-Adapter FaceID | `add_ip_adapter_faceid_to_graph()` + `pipe(..., ip_face_image=...)` |
| Multiple adapters | Add all, then `pipe(..., ip_image=..., ip_image_plus=..., ip_face_image=..., ip_adapter_scale=[...])` |
| `cross_attention_kwargs={"ip_adapter_masks": masks}` | `pipe(..., ip_adapter_masks=masks)` |
| Per-layer (InstantStyle) | `pipe(..., ip_adapter_scale={"down": {"block_2": [0,1]}, "up": {"block_0": [0,1,0]}})` |
