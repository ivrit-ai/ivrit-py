"""
Top‑level package for ivrit.

Exports the public API that users interact with, including model loading
utilities and the newly added word‑level transcription helper.
"""

from .audio import transcribe_word_level
from .utils import load_model  # Assuming load_model lives in utils.py

__all__ = [
    "load_model",
    "transcribe_word_level",
]
