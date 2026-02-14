# yggdrasil/integration/diffusers.py
"""DiffusersBridge — автоматическая конвертация ЛЮБОГО diffusers pipeline в ComputeGraph.

Это главный интеграционный модуль. Позволяет:
1. Загрузить любую модель с HuggingFace и автоматически собрать граф
2. Конвертировать уже инстанцированный diffusers pipeline в ComputeGraph
3. Мапить все diffusers schedulers на YggDrasil solvers
4. Мапить все diffusers models на YggDrasil backbones
"""
from __future__ import annotations

import logging
from typing import Optional, Dict, Any, Type

import torch

from yggdrasil.core.block.builder import BlockBuilder
from yggdrasil.core.block.registry import auto_discover
from yggdrasil.core.model.modular import ModularDiffusionModel
from yggdrasil.core.graph.graph import ComputeGraph

logger = logging.getLogger(__name__)


def _ensure_blocks_registered():
    """Ensure all block modules are imported and registered."""
    auto_discover()


# ==================== DiffusersBridge ====================

class DiffusersBridge:
    """Мост между HuggingFace Diffusers и YggDrasil ComputeGraph.
    
    Автоматически конвертирует ЛЮБОЙ diffusers pipeline в ComputeGraph,
    включая все компоненты: модели, schedulers, encoders, VAE.
    
    Использование::
    
        # Из HuggingFace model ID
        graph = DiffusersBridge.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
        result = graph.execute(prompt="a cat")
        
        # Из инстанса pipeline
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(...)
        graph = DiffusersBridge.import_pipeline(pipe)
    """
    
    # Маппинг всех diffusers schedulers → YggDrasil solvers
    SCHEDULER_MAP: Dict[str, str] = {
        "DDIMScheduler": "diffusion/solver/ddim",
        "DDPMScheduler": "diffusion/solver/ddim",
        "EulerDiscreteScheduler": "solver/euler",
        "EulerAncestralDiscreteScheduler": "solver/euler_ancestral",
        "DPMSolverMultistepScheduler": "solver/dpm_pp_2m",
        "DPMSolverSinglestepScheduler": "solver/dpm_pp_2m",
        "DPMSolverSDEScheduler": "solver/dpm_pp_sde",
        "HeunDiscreteScheduler": "diffusion/solver/heun",
        "FlowMatchEulerDiscreteScheduler": "solver/flow_euler",
        "PNDMScheduler": "solver/pndm",
        "LMSDiscreteScheduler": "solver/lms",
        "UniPCMultistepScheduler": "solver/unipc",
        "DEISMultistepScheduler": "solver/deis",
        "KDPM2DiscreteScheduler": "solver/euler",
        "KDPM2AncestralDiscreteScheduler": "solver/euler_ancestral",
        "EDMDPMSolverMultistepScheduler": "solver/dpm_pp_2m",
        "EDMEulerScheduler": "solver/euler",
        "TCDScheduler": "solver/euler",
        "LCMScheduler": "solver/euler",
        "SASolverScheduler": "solver/euler",
        "DPMSolverMultistepInverseScheduler": "solver/dpm_pp_2m",
        "ScoreSdeVeScheduler": "solver/euler",
        "IPNDMScheduler": "solver/pndm",
        "VQDiffusionScheduler": "diffusion/solver/ddim",
    }
    
    # Маппинг model architectures → YggDrasil backbones
    MODEL_MAP: Dict[str, str] = {
        "UNet2DConditionModel": "backbone/unet2d_condition",
        "UNet2DModel": "backbone/unet2d_condition",
        "UNet3DConditionModel": "backbone/unet3d_condition",
        "UNet1DModel": "backbone/unet2d_condition",  # 1D audio UNet, same wrapper API
        "FluxTransformer2DModel": "backbone/flux_transformer",
        "SD3Transformer2DModel": "backbone/sd3_transformer",
        "DiTTransformer2DModel": "backbone/dit",
        "Transformer2DModel": "backbone/transformer_2d",
        "PixArtTransformer2DModel": "backbone/dit",
        "CogVideoXTransformer3DModel": "backbone/dit",
        "HunyuanDiT2DModel": "backbone/mmdit",
        "LatteTransformer3DModel": "backbone/dit",
        "UNetSpatioTemporalConditionModel": "backbone/unet3d_condition",
        "StableAudioDiTModel": "backbone/dit",  # Stable Audio transformer
    }
    
    # Pipeline type → template name mapping
    PIPELINE_MAP: Dict[str, str] = {
        "StableDiffusionPipeline": "sd15_txt2img",
        "StableDiffusionImg2ImgPipeline": "sd15_img2img",
        "StableDiffusionInpaintPipeline": "sd15_inpainting",
        "StableDiffusionXLPipeline": "sdxl_txt2img",
        "StableDiffusionXLImg2ImgPipeline": "sdxl_img2img",
        "StableDiffusionXLInpaintPipeline": "sdxl_inpainting",
        "StableDiffusion3Pipeline": "sd3_txt2img",
        "FluxPipeline": "flux_txt2img",
        "StableDiffusionControlNetPipeline": "controlnet_txt2img",
        "StableDiffusionXLControlNetPipeline": "controlnet_txt2img",
        "AnimateDiffPipeline": "animatediff_txt2vid",
        "CogVideoXPipeline": "cogvideox_txt2vid",
        "StableVideoDiffusionPipeline": "stable_video_diffusion",
        "I2VGenXLPipeline": "i2vgen_xl",
        "AudioLDMPipeline": "audioldm_txt2audio",
        "AudioLDM2Pipeline": "audioldm2_txt2audio",
        "StableAudioPipeline": "stable_audio",
        "MusicLDMPipeline": "musicldm_txt2audio",
        "DanceDiffusionPipeline": "dance_diffusion_audio",
        "ShapEPipeline": "shap_e_txt2_3d",
        "KandinskyV22Pipeline": "kandinsky_txt2img",
        "IFPipeline": "deepfloyd_txt2img",
        "PixArtAlphaPipeline": "pixart_txt2img",
        "PixArtSigmaPipeline": "pixart_txt2img",
        "StableCascadeDecoderPipeline": "stable_cascade",
    }
    
    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        torch_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> ComputeGraph:
        """Загрузить любую модель с HuggingFace и собрать ComputeGraph.
        
        Автоматически определяет архитектуру и создаёт граф.
        
        Args:
            model_id: HuggingFace model ID или локальный путь.
            torch_dtype: Тип данных (float16 для инференса).
            **kwargs: Доп. параметры.
        
        Returns:
            Готовый ComputeGraph.
        """
        _ensure_blocks_registered()
        
        # Try to load with diffusers first
        try:
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch_dtype, **kwargs
            )
            return cls.import_pipeline(pipe)
        except Exception as e:
            logger.warning(f"Could not load as diffusers pipeline: {e}")
        
        # Fallback: detect from model ID and use templates
        model_id_lower = model_id.lower()
        
        if "flux" in model_id_lower:
            template = "flux_txt2img"
        elif "stable-diffusion-3" in model_id_lower or "sd3" in model_id_lower:
            template = "sd3_txt2img"
        elif "xl" in model_id_lower or "sdxl" in model_id_lower:
            template = "sdxl_txt2img"
        elif "cascade" in model_id_lower:
            template = "stable_cascade"
        elif "kandinsky" in model_id_lower:
            template = "kandinsky_txt2img"
        elif "pixart" in model_id_lower:
            template = "pixart_txt2img"
        elif "audioldm2" in model_id_lower:
            template = "audioldm2_txt2audio"
        elif "audioldm" in model_id_lower:
            template = "audioldm_txt2audio"
        elif "musicldm" in model_id_lower:
            template = "musicldm_txt2audio"
        elif "stable-audio" in model_id_lower or "stable_audio" in model_id_lower:
            template = "stable_audio"
        elif "dance" in model_id_lower and "diffusion" in model_id_lower:
            template = "dance_diffusion_audio"
        elif "animatediff" in model_id_lower:
            template = "animatediff_txt2vid"
        elif "cogvideo" in model_id_lower:
            template = "cogvideox_txt2vid"
        elif "shap-e" in model_id_lower:
            template = "shap_e_txt2_3d"
        else:
            template = "sd15_txt2img"
        
        logger.info(f"Using template '{template}' for model ID '{model_id}'")
        return ComputeGraph.from_template(template, pretrained=model_id, **kwargs)
    
    @classmethod
    def import_pipeline(cls, pipe: Any) -> ComputeGraph:
        """Конвертировать инстанс diffusers pipeline в ComputeGraph.
        
        Инспектирует pipeline и извлекает компоненты.
        """
        _ensure_blocks_registered()
        
        pipe_class = type(pipe).__name__
        logger.info(f"Importing diffusers pipeline: {pipe_class}")
        
        graph = ComputeGraph(f"diffusers_{pipe_class}")
        
        # 1. Extract and wrap backbone model
        backbone_model = cls._find_backbone(pipe)
        if backbone_model is not None:
            backbone_type = cls.MODEL_MAP.get(type(backbone_model).__name__, "backbone/unet2d_condition")
            # Wrap the model directly
            wrapper = _DiffusersModelWrapper(backbone_model, backbone_type)
            graph.add_node("backbone", wrapper)
        
        # 2. Extract VAE
        if hasattr(pipe, 'vae') and pipe.vae is not None:
            vae_wrapper = _DiffusersVAEWrapper(pipe.vae)
            graph.add_node("codec", vae_wrapper)
        
        # 3. Extract text encoders
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            te_wrapper = _DiffusersTextEncoderWrapper(
                pipe.text_encoder,
                getattr(pipe, 'tokenizer', None),
            )
            graph.add_node("conditioner_0", te_wrapper)
            graph.expose_input("prompt", "conditioner_0", "raw_condition")
            graph.connect("conditioner_0", "embedding", "backbone", "condition")
        
        if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None:
            te2_wrapper = _DiffusersTextEncoderWrapper(
                pipe.text_encoder_2,
                getattr(pipe, 'tokenizer_2', None),
            )
            graph.add_node("conditioner_1", te2_wrapper)
            graph.expose_input("prompt_2", "conditioner_1", "raw_condition")
            graph.connect("conditioner_1", "embedding", "backbone", "condition")
        
        # 4. Extract scheduler
        if hasattr(pipe, 'scheduler') and pipe.scheduler is not None:
            scheduler_class = type(pipe.scheduler).__name__
            solver_type = cls.SCHEDULER_MAP.get(scheduler_class, "diffusion/solver/ddim")
            try:
                solver = BlockBuilder.build({"type": solver_type})
                graph.add_node("solver", solver)
            except Exception:
                solver = BlockBuilder.build({"type": "diffusion/solver/ddim"})
                graph.add_node("solver", solver)
        
        # 5. Extract ControlNet if present
        if hasattr(pipe, 'controlnet') and pipe.controlnet is not None:
            cn_wrapper = _DiffusersModelWrapper(pipe.controlnet, "adapter/controlnet")
            graph.add_node("controlnet", cn_wrapper)
            graph.expose_input("control_image", "controlnet", "input")
            graph.connect("controlnet", "output", "backbone", "adapter_features")
        
        # 6. Wire standard graph
        if "backbone" in graph.nodes:
            graph.expose_input("latents", "backbone", "x")
            graph.expose_input("timestep", "backbone", "timestep")
            
            if "solver" in graph.nodes:
                graph.connect("backbone", "output", "solver", "model_output")
                graph.expose_output("next_latents", "solver", "next_latents")
            else:
                graph.expose_output("output", "backbone", "output")
        
        if "codec" in graph.nodes:
            if "solver" in graph.nodes:
                graph.connect("solver", "next_latents", "codec", "latent")
            graph.expose_output("decoded", "codec", "decoded")
        
        logger.info(f"Graph assembled: {graph}")
        return graph
    
    @classmethod
    def _find_backbone(cls, pipe) -> Any:
        """Find the main denoising model in a diffusers pipeline."""
        for attr in ['unet', 'transformer', 'prior', 'decoder']:
            model = getattr(pipe, attr, None)
            if model is not None:
                return model
        return None
    
    @classmethod
    def get_solver_type(cls, scheduler_class_name: str) -> str:
        """Get YggDrasil solver type for a diffusers scheduler."""
        return cls.SCHEDULER_MAP.get(scheduler_class_name, "diffusion/solver/ddim")
    
    @classmethod
    def get_backbone_type(cls, model_class_name: str) -> str:
        """Get YggDrasil backbone type for a diffusers model."""
        return cls.MODEL_MAP.get(model_class_name, "backbone/unet2d_condition")


# ==================== Wrappers ====================

class _DiffusersModelWrapper:
    """Wraps a diffusers model to act as an AbstractBlock-like object."""
    
    def __init__(self, model, block_type: str):
        self._model = model
        self.block_type = block_type
        self.block_id = f"diffusers_{type(model).__name__}"
        self.config = {}
    
    @classmethod
    def declare_io(cls):
        from yggdrasil.core.block.port import InputPort, OutputPort, TensorSpec
        return {
            "x": InputPort("x", spec=TensorSpec(space="latent")),
            "timestep": InputPort("timestep", data_type="tensor"),
            "condition": InputPort("condition", data_type="dict", optional=True),
            "adapter_features": InputPort("adapter_features", data_type="any", optional=True),
            "output": OutputPort("output", spec=TensorSpec(space="latent")),
        }
    
    def process(self, **port_inputs):
        x = port_inputs.get("x")
        timestep = port_inputs.get("timestep")
        condition = port_inputs.get("condition", {})
        encoder_hidden_states = condition.get("encoder_hidden_states") if isinstance(condition, dict) else None
        
        try:
            result = self._model(
                sample=x, timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
        except TypeError:
            try:
                result = self._model(
                    hidden_states=x, timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]
            except Exception:
                result = x
        
        return {"output": result}
    
    def parameters(self):
        return self._model.parameters()
    
    def __repr__(self):
        return f"<DiffusersWrapper {self.block_type}: {type(self._model).__name__}>"


class _DiffusersVAEWrapper:
    """Wraps a diffusers VAE."""
    
    def __init__(self, vae):
        self._vae = vae
        self.block_type = "codec/autoencoder_kl"
        self.block_id = "diffusers_vae"
        self.config = {}
        self.scaling_factor = getattr(vae.config, 'scaling_factor', 0.18215)
    
    @classmethod
    def declare_io(cls):
        from yggdrasil.core.block.port import InputPort, OutputPort, TensorSpec
        return {
            "pixel_data": InputPort("pixel_data", spec=TensorSpec(space="pixel"), optional=True),
            "latent": InputPort("latent", spec=TensorSpec(space="latent"), optional=True),
            "encoded": OutputPort("encoded", spec=TensorSpec(space="latent")),
            "decoded": OutputPort("decoded", spec=TensorSpec(space="pixel")),
        }
    
    def process(self, **port_inputs):
        if "latent" in port_inputs and port_inputs["latent"] is not None:
            z = port_inputs["latent"] / self.scaling_factor
            z = z.to(dtype=self._vae.dtype)
            decoded = self._vae.decode(z).sample
            return {"decoded": decoded, "output": decoded}
        elif "pixel_data" in port_inputs and port_inputs["pixel_data"] is not None:
            x = port_inputs["pixel_data"]
            x = x * 2.0 - 1.0
            z = self._vae.encode(x).latent_dist.sample() * self.scaling_factor
            return {"encoded": z, "output": z}
        return {}
    
    def encode(self, x):
        x = x * 2.0 - 1.0
        return self._vae.encode(x).latent_dist.sample() * self.scaling_factor
    
    def decode(self, z):
        z = z / self.scaling_factor
        z = z.to(dtype=self._vae.dtype)
        return self._vae.decode(z).sample
    
    def parameters(self):
        return self._vae.parameters()


class _DiffusersTextEncoderWrapper:
    """Wraps a diffusers text encoder + tokenizer."""
    
    def __init__(self, encoder, tokenizer):
        self._encoder = encoder
        self._tokenizer = tokenizer
        self.block_type = "conditioner/clip_text"
        self.block_id = f"diffusers_{type(encoder).__name__}"
        self.config = {}
    
    @classmethod
    def declare_io(cls):
        from yggdrasil.core.block.port import InputPort, OutputPort, TensorSpec
        return {
            "raw_condition": InputPort("raw_condition", data_type="dict"),
            "embedding": OutputPort("embedding", spec=TensorSpec(space="embedding")),
        }
    
    def process(self, **port_inputs):
        raw = port_inputs.get("raw_condition", {})
        prompt = raw.get("text", raw.get("prompt", "")) if isinstance(raw, dict) else str(raw)
        if isinstance(prompt, str):
            prompt = [prompt]
        
        if self._tokenizer is not None:
            max_length = getattr(self._tokenizer, 'model_max_length', 77)
            inputs = self._tokenizer(prompt, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self._encoder.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                outputs = self._encoder(**inputs)
            emb = outputs[0]
        else:
            emb = torch.zeros(len(prompt), 77, 768)
        
        return {
            "embedding": emb,
            "encoder_hidden_states": emb,
            "output": emb,
        }
    
    def __call__(self, condition):
        return self.process(raw_condition=condition)
    
    def parameters(self):
        return self._encoder.parameters()


# ==================== Legacy API (backward compatible) ====================

def load_from_diffusers(
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
    torch_dtype: torch.dtype = torch.float16,
    **kwargs,
) -> ModularDiffusionModel:
    """Auto-detect architecture and load as YggDrasil model (legacy API)."""
    _ensure_blocks_registered()
    model_id = pretrained_model_name_or_path.lower()
    
    if "flux" in model_id:
        return load_flux(pretrained_model_name_or_path, torch_dtype=torch_dtype, **kwargs)
    elif "stable-diffusion-3" in model_id or "sd3" in model_id:
        return load_sd3(pretrained_model_name_or_path, torch_dtype=torch_dtype, **kwargs)
    elif "xl" in model_id or "sdxl" in model_id:
        return load_sdxl(pretrained_model_name_or_path, torch_dtype=torch_dtype, **kwargs)
    else:
        return load_stable_diffusion_15(pretrained_model_name_or_path, torch_dtype=torch_dtype, **kwargs)


def load_stable_diffusion_15(pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, **kwargs):
    _ensure_blocks_registered()
    fp16 = torch_dtype == torch.float16
    config = {
        "type": "model/modular", "id": "sd15",
        "backbone": {"type": "backbone/unet2d_condition", "pretrained": pretrained_model_name_or_path, "fp16": fp16},
        "codec": {"type": "codec/autoencoder_kl", "pretrained": pretrained_model_name_or_path, "fp16": fp16, "scaling_factor": 0.18215, "latent_channels": 4, "spatial_scale_factor": 8},
        "conditioner": {"type": "conditioner/clip_text", "pretrained": pretrained_model_name_or_path, "tokenizer_subfolder": "tokenizer", "text_encoder_subfolder": "text_encoder", "max_length": 77},
        "guidance": {"type": "guidance/cfg", "scale": 7.5},
    }
    return BlockBuilder.build(config)


def load_sdxl(pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, **kwargs):
    _ensure_blocks_registered()
    fp16 = torch_dtype == torch.float16
    config = {
        "type": "model/modular", "id": "sdxl",
        "backbone": {"type": "backbone/unet2d_condition", "pretrained": pretrained_model_name_or_path, "fp16": fp16},
        "codec": {"type": "codec/autoencoder_kl", "pretrained": pretrained_model_name_or_path, "fp16": fp16, "scaling_factor": 0.13025, "latent_channels": 4, "spatial_scale_factor": 8},
        "conditioner": {"type": "conditioner/clip_text", "pretrained": pretrained_model_name_or_path, "tokenizer_subfolder": "tokenizer", "text_encoder_subfolder": "text_encoder", "max_length": 77},
        "guidance": {"type": "guidance/cfg", "scale": 7.5},
    }
    return BlockBuilder.build(config)


def load_sd3(pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium", torch_dtype=torch.float16, **kwargs):
    _ensure_blocks_registered()
    fp16 = torch_dtype == torch.float16
    config = {
        "type": "model/modular", "id": "sd3",
        "backbone": {"type": "backbone/mmdit", "hidden_dim": 1536, "num_layers": 24, "num_heads": 24, "in_channels": 16, "patch_size": 2},
        "codec": {"type": "codec/autoencoder_kl", "pretrained": pretrained_model_name_or_path, "fp16": fp16, "latent_channels": 16, "spatial_scale_factor": 8},
        "conditioner": {"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14", "max_length": 77},
        "guidance": {"type": "guidance/cfg", "scale": 5.0},
    }
    return BlockBuilder.build(config)


def load_flux(pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16, **kwargs):
    _ensure_blocks_registered()
    fp16 = torch_dtype == torch.float16
    config = {
        "type": "model/modular", "id": "flux",
        "backbone": {"type": "backbone/mmdit", "hidden_dim": 3072, "num_layers": 19, "num_heads": 24, "in_channels": 16, "patch_size": 2, "cond_dim": 4096},
        "codec": {"type": "codec/autoencoder_kl", "pretrained": pretrained_model_name_or_path, "fp16": fp16, "latent_channels": 16, "spatial_scale_factor": 8},
        "conditioner": {"type": "conditioner/clip_text", "pretrained": "openai/clip-vit-large-patch14", "max_length": 77},
        "guidance": {"type": "guidance/cfg", "scale": 3.5},
    }
    return BlockBuilder.build(config)


def from_diffusers_pipeline(pipe) -> ModularDiffusionModel:
    """Legacy: convert diffusers pipeline to ModularDiffusionModel."""
    from yggdrasil.assemblers.model_assembler import ModelAssembler
    return ModelAssembler.from_diffusers(pipe)
