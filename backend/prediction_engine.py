"""Compatibility shim for legacy imports.

Use backend.prediction.realtime_prediction_engine instead.
"""

from backend.prediction.realtime_prediction_engine import PredictionEngine

__all__ = ["PredictionEngine"]
