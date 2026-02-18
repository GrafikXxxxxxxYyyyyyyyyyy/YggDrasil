"""Time series diffusion plugin."""
from __future__ import annotations

from typing import Dict, Any

from ...plugins.base import AbstractPlugin


class TimeseriesPlugin(AbstractPlugin):
    """Time series generation plugin.
    
    Uses 1D diffusion for time series forecasting, imputation,
    and unconditional generation.
    """
    
    name = "timeseries"
    modality = "timeseries"
    description = "Time series generation, forecasting, and imputation"
    version = "1.0.0"
    
    default_config = {
        "type": "model/modular",
        "backbone": {
            "type": "backbone/transformer_1d",
            "in_channels": 1,
            "hidden_dim": 256,
            "num_layers": 6,
        },
        "codec": {
            "type": "codec/identity",
        },
        "diffusion_process": {"type": "diffusion/process/ddpm"},
    }
    
    @classmethod
    def register_blocks(cls):
        pass
    
    @classmethod
    def get_ui_schema(cls) -> Dict[str, Any]:
        return {
            "inputs": [
                {"type": "text", "name": "prompt", "label": "Description (optional)",
                 "placeholder": "Stock price forecast for next 30 days", "optional": True},
                {"type": "file", "name": "input_csv", "label": "Input CSV",
                 "optional": True},
                {"type": "dataframe", "name": "context", "label": "Context Window",
                 "optional": True},
            ],
            "outputs": [
                {"type": "plot", "name": "result", "label": "Generated Time Series"},
                {"type": "file", "name": "output_csv", "label": "Output CSV"},
            ],
            "advanced": [
                {"type": "slider", "name": "num_steps", "label": "Steps",
                 "min": 1, "max": 200, "default": 100},
                {"type": "slider", "name": "forecast_horizon", "label": "Forecast Horizon",
                 "min": 1, "max": 1000, "default": 100},
                {"type": "slider", "name": "num_channels", "label": "Channels",
                 "min": 1, "max": 100, "default": 1},
                {"type": "dropdown", "name": "task", "label": "Task",
                 "options": ["generation", "forecasting", "imputation"]},
            ],
        }
