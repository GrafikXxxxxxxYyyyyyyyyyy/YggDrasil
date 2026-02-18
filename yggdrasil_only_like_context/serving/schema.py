# yggdrasil/serving/schema.py
"""Pydantic-схемы для API и Gradio.

Единый контракт для любой модальности и любой модели.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum


# ==================== ENUMS ====================

class OutputFormat(str, Enum):
    """Формат вывода."""
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"
    WAV = "wav"
    MP4 = "mp4"
    NPY = "npy"         # Numpy array (для молекул, временных рядов)
    PT = "pt"            # PyTorch tensor
    RAW = "raw"          # Raw bytes
    JSON = "json"        # JSON (для структурированных данных)


class ModelStatus(str, Enum):
    """Статус модели."""
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    UNLOADED = "unloaded"


# ==================== REQUEST ====================

class GenerateRequest(BaseModel):
    """Универсальный запрос на генерацию.
    
    Работает с любой модальностью — все условия передаются через condition.
    """
    # Условия генерации (произвольные: text, image, audio, class_id, ...)
    condition: Dict[str, Any] = Field(default_factory=dict, description="Условия генерации: {text: ..., image: ..., control: ...}")
    
    # Параметры генерации
    num_inference_steps: int = Field(default=28, ge=1, le=1000, description="Число шагов диффузии")
    guidance_scale: float = Field(default=7.5, ge=0.0, le=100.0, description="Guidance scale (CFG)")
    seed: Optional[int] = Field(default=None, description="Seed для воспроизводимости")
    
    # Размер выхода (зависит от модальности)
    width: Optional[int] = Field(default=None, description="Ширина (для изображений/видео)")
    height: Optional[int] = Field(default=None, description="Высота (для изображений/видео)")
    duration: Optional[float] = Field(default=None, description="Длительность (для аудио/видео)")
    num_frames: Optional[int] = Field(default=None, description="Число кадров (для видео)")
    
    # Формат вывода
    output_format: OutputFormat = Field(default=OutputFormat.PNG)
    
    # Кастомные параметры (для расширений и кастомных моделей)
    extra: Dict[str, Any] = Field(default_factory=dict, description="Дополнительные параметры")
    
    # Batch
    batch_size: int = Field(default=1, ge=1, le=64)
    
    # Negative condition
    negative_condition: Optional[Dict[str, Any]] = Field(default=None, description="Негативное условие (negative prompt и т.д.)")
    
    def to_shape(self, default_channels: int = 4, default_size: int = 64) -> tuple:
        """Конвертировать параметры в shape для латентов."""
        h = (self.height or 512) // 8
        w = (self.width or 512) // 8
        return (self.batch_size, default_channels, h, w)


class TrainRequest(BaseModel):
    """Запрос на обучение."""
    model_config = {"protected_namespaces": ()}
    
    model_id: str = Field(description="ID модели для обучения")
    dataset_path: str = Field(description="Путь к данным")
    config: Dict[str, Any] = Field(default_factory=dict, description="Конфигурация обучения")


# ==================== RESPONSE ====================

class GenerateResponse(BaseModel):
    """Ответ генерации."""
    # Результат (base64 или путь к файлу)
    data: Optional[str] = Field(default=None, description="Base64-encoded результат")
    file_path: Optional[str] = Field(default=None, description="Путь к файлу результата")
    
    # Метаданные
    format: OutputFormat = OutputFormat.PNG
    seed: int = 0
    num_steps: int = 0
    generation_time: float = 0.0  # секунды
    
    # Для streaming
    is_partial: bool = False
    step_index: int = 0
    total_steps: int = 0


class TrainResponse(BaseModel):
    """Ответ обучения."""
    status: str = "started"
    job_id: str = ""
    message: str = ""


# ==================== MODEL INFO ====================

class ModelInfo(BaseModel):
    """Информация о загруженной модели."""
    model_config = {"protected_namespaces": ()}
    
    model_id: str
    model_type: str = "unknown"
    modality: str = "unknown"           # "image", "audio", "video", "molecular", ...
    status: ModelStatus = ModelStatus.UNLOADED
    
    # Структура модели
    blocks: Dict[str, str] = Field(default_factory=dict)   # slot_name → block_type
    adapters: List[str] = Field(default_factory=list)
    
    # Возможности
    supports_cfg: bool = True
    supports_streaming: bool = True
    supports_training: bool = True
    
    # Метаданные
    num_parameters: int = 0
    device: str = "cpu"
    dtype: str = "float32"


class ServerStatus(BaseModel):
    """Статус сервера."""
    status: str = "ok"
    version: str = "0.2.0"
    loaded_models: List[ModelInfo] = Field(default_factory=list)
    device: str = "cpu"
    gpu_memory_used: Optional[float] = None    # GB
    gpu_memory_total: Optional[float] = None   # GB


# ==================== CONFIG ====================

class ServerConfig(BaseModel):
    """Конфигурация сервера."""
    model_config = {"protected_namespaces": ()}
    
    host: str = "0.0.0.0"
    port: int = 7860
    api_port: int = 8000
    
    # Модели для загрузки при старте
    preload_models: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Ограничения
    max_batch_size: int = 8
    max_queue_size: int = 100
    timeout: float = 300.0       # секунды
    
    # Безопасность
    api_key: Optional[str] = None
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    
    # Пути
    model_cache_dir: str = "models"
    output_dir: str = "outputs"
