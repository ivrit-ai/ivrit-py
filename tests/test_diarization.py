"""
Test speaker diarization functionality.

This test covers:
1. Two transcription engines: faster-whisper, runpod
2. Two diarization engines: pyannote, ivrit
"""

import os
import pytest
from pathlib import Path

from ivrit import load_model


class TestDiarization:
    """Test speaker diarization functionality"""
    
    @pytest.fixture
    def audio_file_path(self):
        """Get the path to the asimov.mp3 test file"""
        test_dir = Path(__file__).parent
        audio_path = test_dir / "asimov.mp3"
        assert audio_path.exists(), f"Test audio file not found: {audio_path}"
        return str(audio_path)
    
    def test_diarization_core(self, model, diarization_engine, audio_file_path):
        """Core diarization test function"""
        result = model.transcribe(
            path=audio_file_path,
            language='he',
            diarize=True,
            diarization_args={"engine": diarization_engine}
        )
        
        segments = result['segments']
        assert len(segments) > 0
    
    def test_faster_whisper_pyannote(self, audio_file_path):
        """Test faster-whisper with pyannote diarization"""
        model = load_model(engine="faster-whisper", model="ivrit-ai/whisper-large-v3-turbo-ct2")
        self.test_diarization_core(model, "pyannote", audio_file_path)
    
    def test_faster_whisper_ivrit(self, audio_file_path):
        """Test faster-whisper with ivrit diarization"""
        model = load_model(engine="faster-whisper", model="ivrit-ai/whisper-large-v3-turbo-ct2")
        self.test_diarization_core(model, "ivrit", audio_file_path)
    
    def test_runpod_pyannote(self, audio_file_path):
        """Test runpod with pyannote diarization"""
        api_key = os.getenv("RUNPOD_API_KEY")
        endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
        
        if not api_key or not endpoint_id:
            pytest.skip("RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID environment variables required")
        
        model = load_model(
            engine="runpod",
            model="ivrit-ai/whisper-large-v3-turbo-ct2",
            api_key=api_key,
            endpoint_id=endpoint_id
        )
        self.test_diarization_core(model, "pyannote", audio_file_path)
    
    def test_runpod_ivrit(self, audio_file_path):
        """Test runpod with ivrit diarization"""
        api_key = os.getenv("RUNPOD_API_KEY")
        endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
        
        if not api_key or not endpoint_id:
            pytest.skip("RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID environment variables required")
        
        model = load_model(
            engine="runpod",
            model="ivrit-ai/whisper-large-v3-turbo-ct2",
            api_key=api_key,
            endpoint_id=endpoint_id
        )
        self.test_diarization_core(model, "ivrit", audio_file_path)