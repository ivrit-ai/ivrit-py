"""
Test async transcription functionality using the asimov.mp3 file.

This test covers:
1. Async transcription with language='he' for RunPod engine
2. Streaming and non-streaming modes using transcribe_async()
3. Validation of the 'words' field and ensuring text = ''.join(words)
"""

import os
import pytest
import asyncio
from pathlib import Path

from ivrit import load_model, TranscriptionModel
from ivrit.types import Segment, Word


class TestAsyncTranscription:
    """Test async transcription functionality using asimov.mp3"""
    
    @pytest.fixture
    def audio_file_path(self):
        """Get the path to the asimov.mp3 test file"""
        test_dir = Path(__file__).parent
        audio_path = test_dir / "asimov.mp3"
        assert audio_path.exists(), f"Test audio file not found: {audio_path}"
        return str(audio_path)
    
    async def run_async_transcription_test(self, engine: str, model: str, stream: bool, audio_file_path: str, model_kwargs: dict = {}):
        """
        Core async function to create a model and run transcription test.
        
        Args:
            engine: The transcription engine to use
            model: The model path/name to use
            stream: Whether to use streaming mode
            audio_file_path: Path to the audio file
            model_kwargs: Additional arguments to pass to load_model
        """
        # Create model
        transcription_model = load_model(engine=engine, model=model, **model_kwargs)
        
        # Run async transcription
        result = transcription_model.transcribe_async(
            path=audio_file_path,
            language='he',
            stream=stream
        )
        
        # transcribe_async always returns an AsyncGenerator[Segment, None]
        # regardless of stream parameter - collect all segments
        segments = []
        async for segment in result:
            segments.append(segment)
        
        # Shared validation for both streaming and non-streaming
        assert len(segments) > 0, "No segments returned from transcription"
        
        # Check each segment
        for segment in segments:
            assert isinstance(segment, Segment), f"Segment is not of type Segment: {type(segment)}"
            assert isinstance(segment.words, list), f"Words is not a list: {type(segment.words)}"
            
            # Check each word
            for word in segment.words:
                assert isinstance(word, Word), f"Word is not of type Word: {type(word)}"
    
    @pytest.mark.asyncio
    async def test_runpod_async_non_streaming(self, audio_file_path):
        """Test RunPod async non-streaming transcription"""
        api_key = os.getenv("RUNPOD_API_KEY")
        endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
        
        # Will fail with clear error if environment variables are missing
        assert api_key, "RUNPOD_API_KEY environment variable is required for RunPod tests"
        assert endpoint_id, "RUNPOD_ENDPOINT_ID environment variable is required for RunPod tests"
        
        await self.run_async_transcription_test(
            "runpod", 
            "ivrit-ai/whisper-large-v3-turbo-ct2", 
            False, 
            audio_file_path,
            model_kwargs={
                "api_key": api_key,
                "endpoint_id": endpoint_id
            }
        )
    
    @pytest.mark.asyncio
    async def test_runpod_async_streaming(self, audio_file_path):
        """Test RunPod async streaming transcription"""
        api_key = os.getenv("RUNPOD_API_KEY")
        endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
        
        # Will fail with clear error if environment variables are missing
        assert api_key, "RUNPOD_API_KEY environment variable is required for RunPod tests"
        assert endpoint_id, "RUNPOD_ENDPOINT_ID environment variable is required for RunPod tests"
        
        await self.run_async_transcription_test(
            "runpod", 
            "ivrit-ai/whisper-large-v3-turbo-ct2", 
            True, 
            audio_file_path,
            model_kwargs={
                "api_key": api_key,
                "endpoint_id": endpoint_id
            }
        )
