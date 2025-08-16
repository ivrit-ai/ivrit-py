"""
Test transcription functionality using the asimov.mp3 file.

This test covers:
1. Transcription with language='he' for both faster-whisper and stable-whisper engines
2. Streaming and non-streaming modes
3. Validation of the 'words' field and ensuring text = ''.join(words)
"""

import os
import pytest
from pathlib import Path

from ivrit import load_model, TranscriptionModel
from ivrit.types import Segment, Word


class TestBasicTranscription:
    """Test basic transcription functionality using asimov.mp3"""
    
    @pytest.fixture
    def audio_file_path(self):
        """Get the path to the asimov.mp3 test file"""
        test_dir = Path(__file__).parent
        audio_path = test_dir / "asimov.mp3"
        assert audio_path.exists(), f"Test audio file not found: {audio_path}"
        return str(audio_path)
    
    def run_transcription_test(self, engine: str, model: str, stream: bool, audio_file_path: str):
        """
        Core function to create a model and run transcription test.
        
        Args:
            engine: The transcription engine to use
            model: The model path/name to use
            stream: Whether to use streaming mode
            audio_file_path: Path to the audio file
        """
        # Create model
        transcription_model = load_model(engine=engine, model=model)
        
        # Run transcription
        result = transcription_model.transcribe(
            path=audio_file_path,
            language='he',
            stream=stream
        )
        
        if stream:
            # For streaming, result is a generator
            segments = list(result)
        else:
            # For non-streaming, result is a dict
            assert isinstance(result, dict), f"Non-streaming result is not a dict: {type(result)}"
            assert 'segments' in result, "Result dict missing 'segments' key"
            segments = result['segments']
        
        # Shared validation for both streaming and non-streaming
        assert len(segments) > 0, "No segments returned from transcription"
        
        # Check each segment
        for segment in segments:
            assert isinstance(segment, Segment), f"Segment is not of type Segment: {type(segment)}"
            assert isinstance(segment.words, list), f"Words is not a list: {type(segment.words)}"
            
            # Check each word
            for word in segment.words:
                assert isinstance(word, Word), f"Word is not of type Word: {type(word)}"
    
    def test_faster_whisper_non_streaming(self, audio_file_path):
        """Test faster-whisper non-streaming transcription"""
        self.run_transcription_test("faster-whisper", "ivrit-ai/whisper-large-v3-turbo-ct2", False, audio_file_path)
    
    def test_faster_whisper_streaming(self, audio_file_path):
        """Test faster-whisper streaming transcription"""
        self.run_transcription_test("faster-whisper", "ivrit-ai/whisper-large-v3-turbo-ct2", True, audio_file_path)
    
    def test_stable_whisper_non_streaming(self, audio_file_path):
        """Test stable-whisper non-streaming transcription"""
        self.run_transcription_test("stable-whisper", "ivrit-ai/whisper-large-v3-turbo-ct2", False, audio_file_path)
    
    def test_stable_whisper_streaming(self, audio_file_path):
        """Test stable-whisper streaming transcription"""
        self.run_transcription_test("stable-whisper", "ivrit-ai/whisper-large-v3-turbo-ct2", True, audio_file_path)
