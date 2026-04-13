"""
ivrit - Python package providing wrappers around ivrit.ai's capabilities
"""
from __future__ import annotations

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("ivrit")
except:
    __version__ = 'dev'

from .audio import load_model, TranscriptionModel, TranscriptionSession, FasterWhisperModel, StableWhisperModel, WhisperCppModel, RunPodModel
from .types import Segment

__all__ = ['load_model', 'TranscriptionModel', 'TranscriptionSession', 'Segment'] 
