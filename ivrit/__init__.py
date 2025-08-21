"""
ivrit - Python package providing wrappers around ivrit.ai's capabilities
"""

from importlib.metadata import version, PackageNotFoundError
__version__ = version("ivrit")

from .audio import load_model, TranscriptionModel, TranscriptionSession, FasterWhisperModel, StableWhisperModel, RunPodModel
from .types import Segment

__all__ = ['load_model', 'TranscriptionModel', 'TranscriptionSession', 'Segment'] 
