"""Time series modality plugin for YggDrasil.

Supports: forecasting, imputation, and generation of time series data.
"""
from .plugin import TimeseriesPlugin

TimeseriesPlugin.register_blocks()
