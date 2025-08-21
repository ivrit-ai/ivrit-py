"""
Test session functionality for incremental transcription using the asimov.mp3 file.

This test covers:
1. Session creation for both faster-whisper and stable-whisper engines
2. Incremental audio processing with append() in 10-second chunks
3. Final flush and result comparison
"""

import os
import pytest
import subprocess
import logging
from pathlib import Path

from ivrit import load_model 
from ivrit.types import Segment, Word


class TestSessionTranscription:
    """Test session-based incremental transcription functionality using asimov.mp3"""
    
    @pytest.fixture
    def audio_file_path(self):
        """Get the path to the asimov.mp3 test file"""
        test_dir = Path(__file__).parent
        audio_path = test_dir / "asimov.mp3"
        assert audio_path.exists(), f"Test audio file not found: {audio_path}"
        return str(audio_path)
    
    def run_session_test(self, engine: str, model: str, audio_file_path: str):
        """
        Core function to create a model and run session test.
        
        Args:
            engine: The transcription engine to use
            model: The model path/name to use
            audio_file_path: Path to the audio file
        """
        # Create model
        transcription_model = load_model(engine=engine, model=model)
        
        # Create session
        session = transcription_model.create_session(
            language='he',
            sample_rate=16000,
            verbose=True
        )
        
        # Convert audio file to raw PCM s16le format for session processing
        try:
            # Use ffmpeg to convert to raw PCM s16le format (mono, 16-bit, 16kHz)
            cmd = [
                "ffmpeg", "-i", audio_file_path, "-f", "s16le", 
                "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", "-"
            ]
            result = subprocess.run(cmd, capture_output=True, check=True)
            audio_data = result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("ffmpeg not available for audio conversion")
        
        # Calculate chunk size for ~10 second increments
        # Assuming 16kHz, mono, 16-bit: 2 bytes per sample, 16000 samples per second
        bytes_per_second = 16000 * 2  # 32000 bytes per second
        target_chunk_duration = 10  # seconds
        chunk_size = bytes_per_second * target_chunk_duration
        
        # Process audio in chunks
        num_chunks = len(audio_data) // chunk_size + (1 if len(audio_data) % chunk_size else 0)


        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(audio_data))
            chunk = audio_data[start_idx:end_idx]
            
            session.append(chunk)
        
        session.flush()

        # Get final results
        all_segments = session.get_all_segments()

        logging.info(f"Final text: {''.join([segment.text for segment in all_segments])}")

        # Basic validation
        assert isinstance(all_segments, list), "Session segments should be a list"
        
        # Check each segment structure if any were generated
        for segment in all_segments:
            assert isinstance(segment, Segment), f"Segment is not of type Segment: {type(segment)}"
            assert hasattr(segment, 'text'), "Segment missing text attribute"
            assert hasattr(segment, 'start'), "Segment missing start attribute"
            assert hasattr(segment, 'end'), "Segment missing end attribute"
            assert isinstance(segment.start, (int, float)), "Segment start should be numeric"
            assert isinstance(segment.end, (int, float)), "Segment end should be numeric"
            assert segment.start >= 0, "Segment start should be non-negative"
            assert segment.end >= segment.start, "Segment end should be >= start"
    
    def test_faster_whisper_session(self, audio_file_path):
        """Test faster-whisper session functionality"""
        self.run_session_test("faster-whisper", "ivrit-ai/whisper-large-v3-turbo-ct2", audio_file_path)
    
    def test_stable_whisper_session(self, audio_file_path):
        """Test stable-whisper session functionality"""
        self.run_session_test("stable-whisper", "ivrit-ai/whisper-large-v3-turbo-ct2", audio_file_path)
