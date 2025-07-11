"""
ivrit - Python package providing wrappers around ivrit.ai's capabilities
"""

__version__ = '0.0.1'

from .audio import load_model, transcribe, TranscriptionModel, Segment, FasterWhisperModel, StableWhisperModel, RunPodModel

__all__ = ['load_model', 'transcribe', 'TranscriptionModel', 'Segment'] 