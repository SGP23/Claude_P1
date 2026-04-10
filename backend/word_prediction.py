"""Compatibility shim for legacy imports.

Use backend.prediction.word_predictor instead.
"""

from backend.prediction.word_predictor import WordPredictor

__all__ = ["WordPredictor"]
