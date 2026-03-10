"""
Additional REST API Routes
──────────────────────────
Endpoints for word prediction, dataset info, and utility functions.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from backend.word_prediction import WordPredictor

router = APIRouter()

# Shared word predictor instance
_word_predictor = None


def get_word_predictor():
    global _word_predictor
    if _word_predictor is None:
        _word_predictor = WordPredictor()
    return _word_predictor


class SuggestRequest(BaseModel):
    sentence: str
    max_suggestions: int = 4


class CompleteRequest(BaseModel):
    sentence: str
    suggestion: str


@router.post("/suggest-words")
async def suggest_words(req: SuggestRequest):
    """
    Get word suggestions for the current incomplete word.

    Example:
        POST /suggest-words {"sentence": "I AM H"}
        → {"suggestions": ["HAPPY", "HAVE", "HAND", "HELP"]}
    """
    predictor = get_word_predictor()
    suggestions = predictor.get_suggestions(req.sentence, req.max_suggestions)
    return {"suggestions": suggestions}


@router.post("/complete-word")
async def complete_word(req: CompleteRequest):
    """
    Replace incomplete word with a suggestion.

    Example:
        POST /complete-word {"sentence": "I AM H", "suggestion": "HELLO"}
        → {"sentence": "I AM HELLO"}
    """
    predictor = get_word_predictor()
    completed = predictor.complete_word(req.sentence, req.suggestion)
    return {"sentence": completed}


@router.get("/groups")
async def get_groups():
    """Return the ASL letter groups used for disambiguation."""
    return {
        "groups": {
            "0": {"name": "Fist variations", "letters": ["A", "E", "M", "N", "S", "T"]},
            "1": {"name": "Open palm", "letters": ["B", "D", "F", "I", "K", "R", "U", "V", "W"]},
            "2": {"name": "Curved shapes", "letters": ["C", "O"]},
            "3": {"name": "Pointing", "letters": ["G", "H"]},
            "4": {"name": "L-shape", "letters": ["L"]},
            "5": {"name": "Downward", "letters": ["P", "Q"]},
            "6": {"name": "Hook", "letters": ["X"]},
            "7": {"name": "Y-shape", "letters": ["Y"]},
        },
        "excluded": ["J", "Z"],
        "reason": "J and Z require motion (dynamic gestures)",
    }
