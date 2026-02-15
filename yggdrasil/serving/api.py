# yggdrasil/serving/api.py
"""FastAPI REST API для YggDrasil.

Полноценный API endpoint для удалённого использования любой диффузионной модели.

Endpoints:
    POST /generate          — генерация (любая модальность)
    POST /generate/stream   — streaming генерация (WebSocket)
    GET  /models            — список моделей
    POST /models/load       — загрузить модель
    POST /models/unload     — выгрузить модель
    GET  /health            — health check
    POST /train/start       — запустить обучение
    GET  /train/status      — статус обучения
"""
from __future__ import annotations

import io
import base64
import time
import uuid
import asyncio
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from .schema import (
    GenerateRequest, GenerateResponse, TrainRequest, TrainResponse,
    ModelInfo, ModelStatus, ServerConfig, ServerStatus, OutputFormat,
)


# ==================== MODEL MANAGER ====================

class ModelManager:
    """Управление загруженными моделями.
    
    Singleton, хранит все загруженные модели и их сэмплеры.
    """
    
    def __init__(self, config: ServerConfig | None = None):
        self.config = config or ServerConfig()
        self.models: Dict[str, Any] = {}              # model_id → ModularDiffusionModel
        self.samplers: Dict[str, Any] = {}             # model_id → DiffusionSampler
        self.model_info: Dict[str, ModelInfo] = {}     # model_id → ModelInfo
        self.training_jobs: Dict[str, Dict] = {}       # job_id → status
        self._lock = asyncio.Lock()
    
    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    async def load_model(
        self,
        model_id: str,
        config: Dict[str, Any] | None = None,
        pretrained: str | None = None,
        **kwargs,
    ) -> ModelInfo:
        """Загрузить модель (из конфига или pretrained)."""
        async with self._lock:
            if model_id in self.models:
                return self.model_info[model_id]
            
            info = ModelInfo(model_id=model_id, status=ModelStatus.LOADING)
            self.model_info[model_id] = info
        
        try:
            device = self._get_device()
            
            if pretrained:
                # Загрузка из pretrained (diffusers integration)
                from ..integration.diffusers import load_from_diffusers
                model = load_from_diffusers(pretrained, **kwargs)
            elif config:
                # Загрузка из конфига
                from ..core.block.builder import BlockBuilder
                model = BlockBuilder.build(config)
            else:
                raise ValueError("Нужен config или pretrained")
            
            model = model.to(device)
            
            # Создаём сэмплер
            from ..core.engine.sampler import DiffusionSampler
            sampler_config = {"num_inference_steps": 28, "guidance_scale": 7.5}
            
            # Подключаем диффузионный процесс если есть
            process_cfg = kwargs.get("diffusion_process", {"type": "diffusion/process/ddpm"})
            solver_cfg = kwargs.get("solver", {"type": "diffusion/solver/ddim", "eta": 0.0})
            sampler_config["diffusion_process"] = process_cfg
            sampler_config["solver"] = solver_cfg
            
            sampler = DiffusionSampler(sampler_config, model=model)
            sampler = sampler.to(device)
            
            async with self._lock:
                self.models[model_id] = model
                self.samplers[model_id] = sampler
                
                # Заполняем info
                info.status = ModelStatus.READY
                info.device = str(device)
                info.num_parameters = sum(p.numel() for p in model.parameters())
                info.blocks = {
                    k: (v.block_type if hasattr(v, "block_type") else type(v).__name__)
                    for k, v in model._slot_children.items()
                    if not isinstance(v, list)
                }
                self.model_info[model_id] = info
            
            return info
            
        except Exception as e:
            async with self._lock:
                info.status = ModelStatus.ERROR
                self.model_info[model_id] = info
            raise RuntimeError(f"Ошибка загрузки модели {model_id}: {e}") from e
    
    async def unload_model(self, model_id: str):
        """Выгрузить модель (освободить память)."""
        async with self._lock:
            if model_id in self.models:
                del self.models[model_id]
                del self.samplers[model_id]
                self.model_info[model_id].status = ModelStatus.UNLOADED
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    async def generate(self, model_id: str, request: GenerateRequest) -> GenerateResponse:
        """Сгенерировать результат."""
        if model_id not in self.samplers:
            raise ValueError(f"Модель {model_id} не загружена")
        
        sampler = self.samplers[model_id]
        model = self.models[model_id]
        device = next(model.parameters()).device
        
        # Seed
        seed = request.seed or int(torch.randint(0, 2**32, (1,)).item())
        device_str = str(device)
        if device_str == "mps":
            generator = None  # MPS не поддерживает Generator
        else:
            generator = torch.Generator(device_str).manual_seed(seed)
        
        # Shape
        shape = request.to_shape()
        
        start_time = time.time()
        
        # Генерация
        with torch.no_grad():
            result_tensor = sampler.sample(
                condition=request.condition,
                shape=shape,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                generator=generator,
            )
        
        gen_time = time.time() - start_time
        
        # Конвертация в нужный формат
        data_b64 = self._tensor_to_base64(result_tensor, request.output_format)
        
        return GenerateResponse(
            data=data_b64,
            format=request.output_format,
            seed=seed,
            num_steps=request.num_inference_steps,
            generation_time=gen_time,
        )
    
    def _tensor_to_base64(self, tensor: torch.Tensor, fmt: OutputFormat) -> str:
        """Конвертировать тензор в base64."""
        if fmt in (OutputFormat.PNG, OutputFormat.JPEG, OutputFormat.WEBP):
            from PIL import Image
            # Нормализация [-1, 1] → [0, 255]
            img = (tensor / 2 + 0.5).clamp(0, 1)
            img = (img * 255).to(torch.uint8).cpu().numpy()
            if img.ndim == 4:
                img = img[0]  # Берём первый из batch
            if img.shape[0] in (1, 3, 4):  # CHW → HWC
                img = img.transpose(1, 2, 0)
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
            
            pil_img = Image.fromarray(img)
            buf = io.BytesIO()
            pil_fmt = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}[fmt.value]
            pil_img.save(buf, format=pil_fmt)
            return base64.b64encode(buf.getvalue()).decode()
        
        elif fmt == OutputFormat.NPY:
            buf = io.BytesIO()
            np.save(buf, tensor.cpu().numpy())
            return base64.b64encode(buf.getvalue()).decode()
        
        elif fmt == OutputFormat.PT:
            buf = io.BytesIO()
            torch.save(tensor.cpu(), buf)
            return base64.b64encode(buf.getvalue()).decode()
        
        else:
            # RAW bytes
            return base64.b64encode(tensor.cpu().numpy().tobytes()).decode()
    
    def get_status(self) -> ServerStatus:
        """Статус сервера."""
        gpu_used = gpu_total = None
        if torch.cuda.is_available():
            gpu_used = torch.cuda.memory_allocated() / 1e9
            props = torch.cuda.get_device_properties(0)
            gpu_total = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
        
        return ServerStatus(
            status="ok",
            loaded_models=list(self.model_info.values()),
            device=str(self._get_device()),
            gpu_memory_used=gpu_used,
            gpu_memory_total=gpu_total,
        )


# ==================== FASTAPI APP ====================

def create_api(config: ServerConfig | None = None) -> "FastAPI":
    """Создать FastAPI приложение."""
    from fastapi import FastAPI, HTTPException, WebSocket, Depends, Header
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    
    config = config or ServerConfig()
    manager = ModelManager(config)
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Preload models
        for model_cfg in config.preload_models:
            model_id = model_cfg.pop("model_id", model_cfg.get("pretrained", "default"))
            try:
                await manager.load_model(model_id, **model_cfg)
                print(f"Предзагружена модель: {model_id}")
            except Exception as e:
                print(f"Ошибка загрузки {model_id}: {e}")
        yield
        # Cleanup
        for model_id in list(manager.models.keys()):
            await manager.unload_model(model_id)
    
    app = FastAPI(
        title="YggDrasil Diffusion API",
        description="Универсальный API для генерации любой модальности любой диффузионной моделью",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Auth dependency
    async def verify_api_key(authorization: Optional[str] = Header(None)):
        if config.api_key and authorization != f"Bearer {config.api_key}":
            raise HTTPException(status_code=401, detail="Invalid API key")
    
    # --- ENDPOINTS ---
    
    @app.get("/health")
    async def health():
        return manager.get_status()
    
    @app.get("/models", response_model=List[ModelInfo])
    async def list_models():
        return list(manager.model_info.values())
    
    @app.post("/models/{model_id}/load", response_model=ModelInfo)
    async def load_model(
        model_id: str,
        config: Optional[Dict[str, Any]] = None,
        pretrained: Optional[str] = None,
    ):
        try:
            info = await manager.load_model(model_id, config=config, pretrained=pretrained)
            return info
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/models/{model_id}/unload")
    async def unload_model(model_id: str):
        await manager.unload_model(model_id)
        return {"status": "unloaded", "model_id": model_id}
    
    @app.post("/generate/{model_id}", response_model=GenerateResponse)
    async def generate(model_id: str, request: GenerateRequest):
        try:
            response = await manager.generate(model_id, request)
            return response
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.websocket("/generate/{model_id}/stream")
    async def generate_stream(websocket: WebSocket, model_id: str):
        """WebSocket streaming — отправляет промежуточные результаты."""
        await websocket.accept()
        
        try:
            request_data = await websocket.receive_json()
            request = GenerateRequest(**request_data)
            
            if model_id not in manager.samplers:
                await websocket.send_json({"error": f"Модель {model_id} не загружена"})
                return
            
            sampler = manager.samplers[model_id]
            
            step = 0
            for partial_result in sampler.sample_iter(
                condition=request.condition,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
            ):
                step += 1
                data_b64 = manager._tensor_to_base64(partial_result, request.output_format)
                await websocket.send_json({
                    "data": data_b64,
                    "step": step,
                    "total_steps": request.num_inference_steps,
                    "is_partial": step < request.num_inference_steps,
                })
            
        except Exception as e:
            await websocket.send_json({"error": str(e)})
        finally:
            await websocket.close()
    
    @app.post("/train/{model_id}/start", response_model=TrainResponse)
    async def start_training(model_id: str, request: TrainRequest):
        """Запустить обучение (асинхронно)."""
        if model_id not in manager.models:
            raise HTTPException(status_code=404, detail=f"Модель {model_id} не загружена")
        
        job_id = str(uuid.uuid4())[:8]
        manager.training_jobs[job_id] = {
            "status": "starting",
            "model_id": model_id,
            "config": request.config,
        }
        
        # Запускаем обучение в фоне
        asyncio.create_task(_run_training(manager, job_id, model_id, request))
        
        return TrainResponse(status="started", job_id=job_id, message=f"Training job {job_id} started")
    
    @app.get("/train/{job_id}/status")
    async def training_status(job_id: str):
        if job_id not in manager.training_jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} не найден")
        return manager.training_jobs[job_id]
    
    # ==================== GRAPH API ====================
    
    @app.post("/graph/execute")
    async def graph_execute(request: dict):
        """Execute a ComputeGraph.
        
        Body::
        
            {
                "template": "sd15_txt2img",  // or provide "graph_yaml"
                "inputs": {
                    "prompt": {"text": "a cat"},
                    "latents": null  // auto-generate noise if null
                }
            }
        """
        from ..core.graph.graph import ComputeGraph
        from ..core.graph.executor import GraphExecutor
        
        template = request.get("template")
        graph_yaml = request.get("graph_yaml")
        inputs = request.get("inputs", {})

        if template:
            graph = ComputeGraph.from_template(template)
        elif graph_yaml:
            import tempfile, os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(graph_yaml)
                graph = ComputeGraph.from_yaml(f.name)
                os.unlink(f.name)
        elif getattr(app.state, "default_graph", None) is not None:
            graph = app.state.default_graph
        else:
            raise HTTPException(status_code=400, detail="Provide 'template', 'graph_yaml', or run with default graph (e.g. Vast.ai deploy)")

        executor = GraphExecutor()
        outputs = executor.execute(graph, **inputs)

        # Serialize outputs (convert tensors to shape/dtype or base64 for decoded image)
        serialized = {}
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                if k == "decoded" and v.dim() >= 3 and request.get("return_decoded_b64"):
                    serialized["decoded_b64"] = manager._tensor_to_base64(v, OutputFormat.PNG)
                serialized[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
            else:
                serialized[k] = str(v)

        return {"outputs": serialized, "graph_name": graph.name}
    
    @app.post("/graph/load")
    async def graph_load(request: dict):
        """Load a graph from YAML string."""
        from ..core.graph.graph import ComputeGraph
        
        yaml_str = request.get("yaml")
        if not yaml_str:
            raise HTTPException(status_code=400, detail="Provide 'yaml' field")
        
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_str)
            graph = ComputeGraph.from_yaml(f.name)
            os.unlink(f.name)
        
        return {
            "name": graph.name,
            "nodes": list(graph.nodes.keys()),
            "edges": [str(e) for e in graph.edges],
            "inputs": list(graph.graph_inputs.keys()),
            "outputs": list(graph.graph_outputs.keys()),
        }
    
    @app.post("/graph/modify")
    async def graph_modify(request: dict):
        """Modify a graph node.
        
        Body::
        
            {
                "template": "sd15_txt2img",
                "operation": "replace_node",  // or "add_node", "remove_node", "connect"
                "node_name": "backbone",
                "block_config": {"type": "backbone/dit", "hidden_dim": 1152}
            }
        """
        from ..core.graph.graph import ComputeGraph
        from ..core.block.builder import BlockBuilder
        
        template = request.get("template")
        graph = ComputeGraph.from_template(template)
        
        operation = request.get("operation")
        node_name = request.get("node_name")
        block_config = request.get("block_config")
        
        if operation == "replace_node":
            block = BlockBuilder.build(block_config)
            graph.replace_node(node_name, block)
        elif operation == "add_node":
            block = BlockBuilder.build(block_config)
            graph.add_node(node_name, block)
        elif operation == "remove_node":
            graph.remove_node(node_name)
        elif operation == "connect":
            src = request.get("src_node")
            src_port = request.get("src_port")
            dst = request.get("dst_node")
            dst_port = request.get("dst_port")
            graph.connect(src, src_port, dst, dst_port)
        
        return {
            "name": graph.name,
            "nodes": list(graph.nodes.keys()),
            "edges": [str(e) for e in graph.edges],
        }
    
    @app.get("/graph/visualize")
    async def graph_visualize(template: str = "sd15_txt2img"):
        """Get Mermaid diagram of a graph."""
        from ..core.graph.graph import ComputeGraph
        graph = ComputeGraph.from_template(template)
        return {"mermaid": graph.visualize(), "name": graph.name}
    
    @app.get("/graph/templates")
    async def graph_templates():
        """List available pipeline templates."""
        from ..core.graph.templates import list_templates
        return {"templates": list_templates()}
    
    @app.post("/graph/train")
    async def graph_train(request: dict):
        """Start training on graph nodes.
        
        Body::
        
            {
                "template": "sd15_txt2img",
                "train_nodes": ["backbone"],
                "dataset_path": "/path/to/data",
                "config": {"num_epochs": 10, "learning_rate": 1e-4}
            }
        """
        job_id = str(uuid.uuid4())[:8]
        manager.training_jobs[job_id] = {
            "status": "starting",
            "type": "graph",
            "template": request.get("template"),
            "train_nodes": request.get("train_nodes"),
        }
        
        asyncio.create_task(_run_graph_training(manager, job_id, request))
        
        return {"status": "started", "job_id": job_id}
    
    # Store manager on app for Gradio access
    app.state.manager = manager
    
    return app


async def _run_training(manager: ModelManager, job_id: str, model_id: str, request: TrainRequest):
    """Фоновая задача обучения."""
    try:
        manager.training_jobs[job_id]["status"] = "running"
        
        from ..training.trainer import DiffusionTrainer, TrainingConfig
        from ..training.data import ImageFolderSource
        from ..core.diffusion.ddpm import DDPMProcess
        from ..training.loss import EpsilonLoss
        
        model = manager.models[model_id]
        process = DDPMProcess()
        loss_fn = EpsilonLoss()
        config = TrainingConfig.from_dict(request.config)
        
        trainer = DiffusionTrainer(model, process, loss_fn, config)
        dataset = ImageFolderSource(request.dataset_path)
        
        # Blocking call — в реальном проекте вынести в отдельный процесс
        loop = asyncio.get_event_loop()
        history = await loop.run_in_executor(None, trainer.train, dataset)
        
        manager.training_jobs[job_id]["status"] = "completed"
        manager.training_jobs[job_id]["history"] = {
            "total_time": history.get("total_time", 0),
            "final_loss": history["loss"][-1] if history.get("loss") else None,
        }
    except Exception as e:
        manager.training_jobs[job_id]["status"] = "error"
        manager.training_jobs[job_id]["error"] = str(e)


async def _run_graph_training(manager: ModelManager, job_id: str, request: dict):
    """Background task for graph-based training."""
    try:
        manager.training_jobs[job_id]["status"] = "running"
        
        from ..core.graph.graph import ComputeGraph
        from ..training.graph_trainer import GraphTrainer, GraphTrainingConfig
        
        template = request.get("template", "sd15_txt2img")
        train_nodes = request.get("train_nodes", [])
        dataset_path = request.get("dataset_path", "")
        config_dict = request.get("config", {})
        
        graph = ComputeGraph.from_template(template)
        
        config = GraphTrainingConfig(
            train_nodes=train_nodes,
            num_epochs=config_dict.get("num_epochs", 10),
            learning_rate=config_dict.get("learning_rate", 1e-4),
            batch_size=config_dict.get("batch_size", 1),
            output_dir=config_dict.get("output_dir", f"./runs/graph_train_{job_id}"),
        )
        
        trainer = GraphTrainer(graph=graph, config=config)
        
        loop = asyncio.get_event_loop()
        history = await loop.run_in_executor(None, trainer.train, dataset_path)
        
        manager.training_jobs[job_id]["status"] = "completed"
        manager.training_jobs[job_id]["history"] = history
    except Exception as e:
        manager.training_jobs[job_id]["status"] = "error"
        manager.training_jobs[job_id]["error"] = str(e)
