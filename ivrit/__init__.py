"""
ivrit - Python package providing wrappers around ivrit.ai's capabilities
"""

__version__ = "0.0.1"

from .audio import FasterWhisperModel, RunPodModel, StableWhisperModel, TranscriptionModel, load_model

__all__ = ["FasterWhisperModel", "RunPodModel", "StableWhisperModel", "TranscriptionModel", "load_model"]
