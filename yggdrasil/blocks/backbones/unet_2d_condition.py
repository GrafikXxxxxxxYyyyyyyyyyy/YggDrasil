import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from omegaconf import DictConfig

from ...core.block.registry import register_block
from ...core.model.backbone import AbstractBackbone


@register_block("backbone/unet2d_condition")
class UNet2DConditionBackbone(AbstractBackbone):
    """Нативная обёртка UNet2DConditionModel из diffusers (SD 1.5 / SDXL)."""
    
    block_type = "backbone/unet2d_condition"
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        pretrained = config.get("pretrained", "runwayml/stable-diffusion-v1-5")
        subfolder = "unet"
        dtype = torch.float16 if config.get("fp16", True) else torch.float32
        raw_config = UNet2DConditionModel.load_config(pretrained, subfolder=subfolder)
        if isinstance(raw_config.get("sample_size"), list):
            raw_config = dict(raw_config)
            raw_config["sample_size"] = tuple(raw_config["sample_size"])
            self.unet = UNet2DConditionModel.from_config(raw_config)
            state = None
            try:
                from diffusers.utils import _get_model_file
                model_file = _get_model_file(pretrained, subfolder=subfolder)
                if str(model_file).endswith(".safetensors"):
                    import safetensors.torch
                    state = safetensors.torch.load_file(model_file, device="cpu")
                else:
                    state = torch.load(model_file, map_location="cpu", weights_only=True)
            except Exception:
                try:
                    from huggingface_hub import hf_hub_download
                    for fname in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"):
                        try:
                            path = hf_hub_download(pretrained, f"{subfolder}/{fname}", repo_type="model")
                            if fname.endswith(".safetensors"):
                                import safetensors.torch
                                state = safetensors.torch.load_file(path, device="cpu")
                            else:
                                state = torch.load(path, map_location="cpu", weights_only=True)
                            break
                        except Exception:
                            continue
                except Exception:
                    pass
            if isinstance(state, dict):
                if "state_dict" in state:
                    state = state["state_dict"]
                self.unet.load_state_dict(state, strict=True)
            self.unet = self.unet.to(dtype=dtype)
        else:
            self.unet = UNet2DConditionModel.from_pretrained(
                pretrained, subfolder=subfolder, torch_dtype=dtype
            )
        self.unet.requires_grad_(False)  # по умолчанию заморожен
    
    def _forward_impl(
        self,
        x: torch.Tensor,                    # [B, 4, 64, 64]
        timestep: torch.Tensor,             # [B]
        condition: dict | None = None,      # {"encoder_hidden_states": [B, 77, 768]}
        **kwargs
    ) -> torch.Tensor:
        encoder_hidden_states = condition.get("encoder_hidden_states") if condition else None
        
        # Determine model device and dtype
        model_device = next(self.unet.parameters()).device
        model_dtype = next(self.unet.parameters()).dtype
        
        # Ensure all inputs are on the same device and dtype as the model
        if x.device != model_device:
            x = x.to(model_device)
        if timestep.device != model_device:
            timestep = timestep.to(model_device)
        if encoder_hidden_states is not None and encoder_hidden_states.device != model_device:
            encoder_hidden_states = encoder_hidden_states.to(model_device)
        # Cast sample and encoder to model dtype; timestep must stay long for diffusers time_embed
        if encoder_hidden_states is not None and encoder_hidden_states.dtype != model_dtype:
            encoder_hidden_states = encoder_hidden_states.to(dtype=model_dtype)
        if x.dtype != model_dtype:
            x = x.to(dtype=model_dtype)
        if timestep.dtype not in (torch.long, torch.int64):
            timestep = timestep.long()
        
        # Pass through ControlNet/Adapter residuals (from condition dict, kwargs, or adapter_features port)
        down_block_residuals = kwargs.get("down_block_additional_residuals")
        mid_block_residual = kwargs.get("mid_block_additional_residual")
        if down_block_residuals is None and condition is not None and isinstance(condition, dict):
            down_block_residuals = condition.get("down_block_additional_residuals")
            mid_block_residual = condition.get("mid_block_additional_residual")
        if down_block_residuals is None:
            af = kwargs.get("adapter_features")
            if isinstance(af, dict):
                down_block_residuals = af.get("down_block_additional_residuals") or af.get("down_block_residuals")
                mid_block_residual = af.get("mid_block_additional_residual") or af.get("mid_block_residual")
            elif isinstance(af, (tuple, list)) and len(af) >= 2 and not (af and isinstance(af[0], dict)):
                down_block_residuals, mid_block_residual = af[0], af[1]
            elif isinstance(af, list) and af and isinstance(af[0], dict):
                # Multiple adapters (ControlNet + T2I): sum residuals element-wise
                all_down = [a.get("down_block_additional_residuals") or a.get("down_block_residuals") for a in af]
                all_down = [x for x in all_down if x is not None]
                all_mid = [a.get("mid_block_additional_residual") or a.get("mid_block_residual") for a in af]
                all_mid = [x for x in all_mid if x is not None]
                if all_down:
                    down_block_residuals = [sum(t) for t in zip(*all_down)]
                if all_mid:
                    mid_block_residual = sum(all_mid)

        added_cond_kwargs = None
        if condition is not None and isinstance(condition, dict):
            added_cond_kwargs = condition.get("added_cond_kwargs")
        if added_cond_kwargs is None:
            added_cond_kwargs = kwargs.get("added_cond_kwargs")

        unet_kw = dict(
            sample=x,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_residuals,
            mid_block_additional_residual=mid_block_residual,
            return_dict=False,
        )
        if added_cond_kwargs is not None:
            unet_kw["added_cond_kwargs"] = added_cond_kwargs
        # AudioLDM and similar: when num_class_embeds > 0 the UNet expects class_labels
        if getattr(self.unet, "class_embedding", None) is not None:
            class_labels = None
            if condition is not None and isinstance(condition, dict):
                class_labels = condition.get("class_labels")
            if class_labels is None:
                batch_size = x.shape[0]
                class_labels = torch.zeros(batch_size, device=model_device, dtype=torch.long)
            elif class_labels.device != model_device:
                class_labels = class_labels.to(model_device)
            unet_kw["class_labels"] = class_labels
        return self.unet(**unet_kw)[0]