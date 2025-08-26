"""
Audio transcription functionality for ivrit.ai
"""
import asyncio
import base64
import json
import os
import time
import io
import wave
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Generator, Optional, Union, List, Dict
from uuid import uuid4

import aiohttp
import requests

from . import utils
from .types import Segment, Word


def _copy_segment_extra_data(segment, language: Optional[str] = None) -> dict:
    """
    Copy extra data from a segment object, filtering out bound methods and other non-value attributes.
    
    Args:
        segment: The segment object to extract data from
        language: Optional language override
    
    Returns:
        Dictionary containing the extra data
    """
    extra_data = {}
    
    # Add all segment attributes to extra_data, filtering out non-serializable attributes
    for attr_name in dir(segment):
        if not attr_name.startswith('_') and attr_name not in ['text', 'start', 'end', 'words']:
            try:
                attr_value = getattr(segment, attr_name)
                # Test if the attribute is serializable by trying to convert to JSON
                json.dumps(attr_value)
                extra_data[attr_name] = attr_value
            except (TypeError, ValueError):
                # Skip non-serializable attributes
                pass
            except Exception:
                # Skip attributes that can't be accessed
                pass
       
    return extra_data


class TranscriptionSession(ABC):
    """
    Abstract base class for incremental transcription sessions.
    
    A session maintains state for incremental audio processing, allowing users
    to add audio frames and get new segments with confidence scores.
    """
    
    def __init__(self, session_id: str, model: 'TranscriptionModel'):
        """
        Initialize a transcription session.
        
        Args:
            session_id: Unique identifier for this session
            model: The transcription model to use
        """
        self.session_id = session_id
        self.model = model
    
    @abstractmethod
    def append(self, audio_bytes: bytes) -> None:
        """
        Add audio to the session and update internal state. Does not return segments.

        The input should be raw mono 16-bit PCM bytes (s16le) at the session's sample_rate.

        Args:
            audio_bytes: Audio payload as raw PCM s16le bytes
        """
        pass
    
    @abstractmethod
    def get_all_segments(self) -> List[Segment]:
        """
        Get all segments accumulated in this session.
        
        Returns:
            List of all segments in the session
        """
        pass
    
    @abstractmethod
    def get_full_text(self) -> str:
        """
        Get the full transcribed text from all segments.
        
        Returns:
            Combined text from all segments
        """
        pass
    
    @abstractmethod
    def get_session_info(self) -> Dict[str, Any]:
        """
        Get information about this session.
        
        Returns:
            Dictionary containing session metadata
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the session buffer and clear all state."""
        pass
    
    @abstractmethod
    def flush(self) -> List[Segment]:
        """
        Flush the session and return any remaining segments including the final one.
        
        This method should be called at the end of audio processing to get the
        final segment(s) that may not have been returned by append() due to
        confidence filtering.
        
        Returns:
            List of remaining segments including the final one
        """
        pass


class TranscriptionModel(ABC):
    """Base class for transcription models"""
    
    def __init__(self, engine: str, model: str, model_object: Any = None):
        self.engine = engine
        self.model = model
        self.model_object = model_object

    def __repr__(self):
        return f"{self.__class__.__name__}(engine='{self.engine}', model='{self.model}')"
    
    def create_session(self, language: Optional[str] = None, sample_rate: int = 16000, 
                      verbose: bool = False) -> TranscriptionSession:
        """
        Create a new transcription session for incremental audio processing.
        
        Args:
            language: Language code for transcription (e.g., 'he' for Hebrew, 'en' for English)
            sample_rate: Audio sample rate (default: 16000 Hz)
            verbose: Whether to enable verbose output
            
        Returns:
            TranscriptionSession object for incremental transcription
            
        Raises:
            NotImplementedError: If the model doesn't support session-based transcription
        """
        raise NotImplementedError(f"Session-based transcription is not supported for {self.engine} engine. "
                                f"Only some specific models support sessions.")
    

    
    def transcribe(
        self,
        *,
        path: Optional[str] = None,
        url: Optional[str] = None,
        blob: Optional[str] = None,
        language: Optional[str] = None,
        stream: bool = False,
        diarize: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Union[dict, Generator]:
        """
        Transcribe audio using this model.
        
        Args:
            path: Path to the audio file to transcribe (mutually exclusive with url and blob)
            url: URL to download and transcribe (mutually exclusive with path and blob)
            blob: Base64 encoded blob data to transcribe (mutually exclusive with path and url)
            language: Language code for transcription (e.g., 'he' for Hebrew, 'en' for English)
            stream: Whether to return results as a generator (True) or full result (False)
            diarize: Whether to enable speaker diarization  
            verbose: Whether to enable verbose output
            **kwargs: Additional keyword arguments for the transcription model.
        Returns:
            If stream=True: Generator yielding transcription segments
            If stream=False: Complete transcription result as dictionary
            
        Raises:
            ValueError: If multiple input sources are provided, or none is provided
            FileNotFoundError: If the specified path doesn't exist
            Exception: For other transcription errors
        """
        # Validate arguments
        provided_args = [arg for arg in [path, url, blob] if arg is not None]
        if len(provided_args) > 1:
            raise ValueError("Cannot specify multiple input sources - path, url, and blob are mutually exclusive")
        
        if len(provided_args) == 0:
            raise ValueError("Must specify either 'path', 'url', or 'blob'")

        # Get streaming results from the model
        segments_generator = self.transcribe_core(path=path, url=url, blob=blob, language=language, diarize=diarize, verbose=verbose, **kwargs)
        
        if stream:
            # Return generator directly
            return segments_generator
        else:
            # Collect all segments and return as dictionary
            segments = list(segments_generator)
            if not segments:
                return {
                    "text": "",
                    "segments": [],
                    "language": language or "unknown",
                    "engine": self.engine,
                    "model": self.model
                }
            
            # Combine all text
            full_text = " ".join(segment.text for segment in segments)
            
            transcription_results = {
                "text": full_text,
                "segments": segments,
                "language": segments[0].extra_data.get("language", language or "unknown"),
                "engine": self.engine,
                "model": self.model
            }

            return transcription_results
    
    @abstractmethod
    def transcribe_core(
        self, 
        *, 
        path: Optional[str] = None,
        url: Optional[str] = None,
        blob: Optional[str] = None,
        language: Optional[str] = None,
        diarize: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Generator[Segment, None, None]:
        """
        Core transcription method that must be implemented by derived classes.
        
        Args:
            path: Path to the audio file to transcribe (mutually exclusive with url and blob)
            url: URL to download and transcribe (mutually exclusive with path and blob)
            blob: Base64 encoded blob data to transcribe (mutually exclusive with path and url)
            language: Language code for transcription
            diarize: Whether to enable speaker diarization
            verbose: Whether to enable verbose output
            **kwargs: Additional keyword arguments for the transcription model.
            
        Returns:
            Generator yielding Segment objects
        """


def get_device_and_index(device: str) -> tuple[str, Optional[int]]:
    """
    Parse device string to extract device type and index.
    
    Args:
        device: Device string (e.g., "cuda", "cuda:0", "cpu")
        
    Returns:
        Tuple of (device_type, device_index)
    """
    if ":" in device:
        device_type, index_str = device.split(":", 1)
        return device_type, int(index_str)
    else:
        return device, None


class WhisperSession(TranscriptionSession):
    """
    Concrete session implementation for transcription models.
    
    Manages incremental audio processing with confidence tracking for whisper-based engines.
    """
    
    def __init__(self, session_id: str, model: TranscriptionModel, language: Optional[str] = None, 
                 sample_rate: int = 16000, verbose: bool = True):
        """
        Initialize a whisper transcription session.
        
        Args:
            session_id: Unique identifier for this session
            model: The TranscriptionModel to use
            language: Language code for transcription
            sample_rate: Audio sample rate (default: 16000 Hz)
            verbose: Whether to enable verbose output
        """
        super().__init__(session_id, model)
        self.language = language
        self.sample_rate = sample_rate
        self.verbose = verbose
        
        # Audio buffer for incremental processing (raw PCM s16le bytes)
        self.pcm_bytes_buffer = bytearray()
        
        # Track accumulated segments
        self.accumulated_segments: List[Segment] = []
        
        # Session metadata
        self.total_frames_added = 0
        self.total_duration = 0.0
    
    def append(self, audio_bytes: bytes) -> None:
        """
        Add audio to the session and update internal state. Does not return segments.

        Accepts raw mono 16-bit PCM bytes (s16le) at the session's sample_rate.
        """
        if not audio_bytes:
            return

        # Validate PCM s16le input
        if len(audio_bytes) % 2 != 0:
            raise ValueError("PCM bytes length must be even (16-bit samples)")
        
        # Add raw PCM s16le to buffer
        self.pcm_bytes_buffer.extend(audio_bytes)
        self.total_frames_added += len(audio_bytes) // 2
        
        # Update duration
        self.total_duration = len(self.pcm_bytes_buffer) / (2 * self.sample_rate)

        if self.verbose:
            print(
                f"Session {self.session_id}: Added {len(audio_bytes)} bytes, "
                f"total duration: {self.total_duration:.2f}s"
            )

        # Process buffer to extract any complete segments and accumulate them
        complete_segments = self._transcribe_buffered(flush=False)
        if complete_segments:
            self.accumulated_segments.extend(complete_segments)
            if self.verbose:
                print(
                    f"Session {self.session_id}: Found {len(complete_segments)} complete segments"
                )
    
    def get_all_segments(self) -> List[Segment]:
        """
        Get all segments accumulated in this session.
        
        Returns:
            List of all segments in the session
        """
        return self.accumulated_segments.copy()
    
    def get_full_text(self) -> str:
        """
        Get the full transcribed text from all segments.
        
        Returns:
            Combined text from all segments
        """
        return " ".join(segment.text for segment in self.accumulated_segments)
    
    def get_session_info(self) -> Dict[str, Any]:
        """
        Get information about this session.
        
        Returns:
            Dictionary containing session metadata
        """
        return {
            "session_id": self.session_id,
            "total_frames": self.total_frames_added,
            "total_duration": self.total_duration,
            "total_segments": len(self.accumulated_segments),
            "sample_rate": self.sample_rate,
            "language": self.language,
            "engine": self.model.engine,
            "model": self.model.model
        }
    
    def reset(self):
        """Reset the session buffer and clear all state."""
        self.pcm_bytes_buffer = bytearray()
        self.accumulated_segments = []
        self.total_frames_added = 0
        self.total_duration = 0.0
        
        if self.verbose:
            print(f"Session {self.session_id}: Reset")
    
    def flush(self) -> List[Segment]:
        """
        Flush the session and return any remaining segments including the final one.
        
        This method should be called at the end of audio processing to get the
        final segment(s) that may not have been returned by append() due to
        confidence filtering.
        
        Returns:
            List of remaining segments including the final one
        """
        if len(self.pcm_bytes_buffer) == 0:
            return []
        
        # Get any remaining segments (handles buffer clearing internally)
        remaining_segments = self._transcribe_buffered(flush=True)
        
        # Add remaining segments to accumulated and return them
        if remaining_segments:
            self.accumulated_segments.extend(remaining_segments)
            
            if self.verbose:
                print(f"Session {self.session_id}: Flushed {len(remaining_segments)} final segments")
            
            return remaining_segments
        
        return []
    
    def _transcribe_buffered(self, flush: bool = False) -> List[Segment]:
        """
        Transcribe the current audio buffer using the model and handle buffer trimming.
        
        Uses the model's transcribe_core method to get Segment objects without looking
        at model internals, providing a unified implementation across all model types.
        
        Args:
            flush: If True, return all segments and clear the buffer.
                   If False, return complete segments and trim the buffer.
        
        Returns:
            List of transcription segments based on flush parameter
        """
        if len(self.pcm_bytes_buffer) == 0:
            return []
        
        # Skip if buffer is too short (less than 0.5 seconds), unless flushing
        min_bytes = int(0.5 * self.sample_rate) * 2  # 16-bit mono => 2 bytes per sample
        if not flush and len(self.pcm_bytes_buffer) < min_bytes:
            return []
        
        # Create in-memory WAV buffer from raw PCM bytes
        wav_buffer = io.BytesIO()
        
        try:
            # Create WAV data in memory
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(bytes(self.pcm_bytes_buffer))
            
            # Get WAV bytes and convert to base64 blob for all models
            # This eliminates temporary file creation and uses utils.get_audio_file_path blob handling
            wav_bytes = wav_buffer.getvalue()
            import base64
            wav_blob = base64.b64encode(wav_bytes).decode('utf-8')
            
            # Use the model's transcribe_core method with blob for all model types
            # This works uniformly since all models support blob via utils.get_audio_file_path
            all_segments = list(self.model.transcribe_core(
                blob=wav_blob,
                language=self.language,
                verbose=self.verbose
            ))
            
            # Handle buffer trimming and segment filtering based on flush parameter
            if flush:
                # When flushing, clear the buffer and return all segments
                self.pcm_bytes_buffer = bytearray()
                return all_segments
            else:
                # For regular processing, return complete segments and trim buffer
                if len(all_segments) > 1:
                    # All segments except the last are considered complete/high-confidence
                    complete_segments = all_segments[:-1]
                    
                    # Trim the buffer: remove audio up to the end of the last complete segment
                    last_complete_end_time = complete_segments[-1].end
                    samples_to_remove = int(last_complete_end_time * self.sample_rate)
                    bytes_to_remove = samples_to_remove * 2
                    
                    if bytes_to_remove > 0 and bytes_to_remove < len(self.pcm_bytes_buffer):
                        self.pcm_bytes_buffer = self.pcm_bytes_buffer[bytes_to_remove:]
                        
                        if self.verbose:
                            print(f"Session {self.session_id}: Trimmed {samples_to_remove} samples ({last_complete_end_time:.2f}s)")
                    
                    return complete_segments
                else:
                    # No complete segments yet (0 or 1 segments)
                    return []
            
        except Exception as e:
            if self.verbose:
                print(f"Error during session buffer transcription: {e}")
            return []
        finally:
            # Close the WAV buffer
            wav_buffer.close()


class FasterWhisperModel(TranscriptionModel):
    """Faster Whisper transcription model"""
    
    def __init__(self, model: str, device: str = None, local_files_only: bool = False):
        super().__init__(engine="faster-whisper", model=model)
        
        self.model_path = model
        self.device = device if device else utils.guess_device()
        self.local_files_only = local_files_only
        
        # Load the model immediately
        self.model_object = self._load_faster_whisper_model()
    
    def _load_faster_whisper_model(self) -> Any:
        """
        Load the actual faster-whisper model.
        """
        # Import faster_whisper
        try:
            import faster_whisper
        except ImportError:
            raise ImportError("faster-whisper is not installed. Please install it with: pip install faster-whisper")
        
        device_index = None
        
        if len(self.device.split(",")) > 1:
            device_indexes = []
            base_device = None
            for device_instance in self.device.split(","):
                device, device_index = get_device_and_index(device_instance)
                base_device = base_device or device
                if base_device != device:
                    raise ValueError("Multiple devices must be instances of the same base device (e.g cuda:0, cuda:1 etc.)")
                device_indexes.append(device_index)
            device = base_device
            device_index = device_indexes
        else:
            device, device_index = get_device_and_index(self.device)
        
        args = {'device': device}
        if device_index:
            args['device_index'] = device_index
        if self.local_files_only:
            args['local_files_only'] = self.local_files_only
        
        print(f'Loading faster-whisper model: {self.model_path} on {device} with index: {device_index or 0}')
        return faster_whisper.WhisperModel(self.model_path, **args)
    
    def create_session(self, language: Optional[str] = None, sample_rate: int = 16000, 
                      verbose: bool = False) -> TranscriptionSession:
        """
        Create a new transcription session for incremental audio processing.
        
        Args:
            language: Language code for transcription (e.g., 'he' for Hebrew, 'en' for English)
            sample_rate: Audio sample rate (default: 16000 Hz)
            verbose: Whether to enable verbose output
            
        Returns:
            WhisperSession object for incremental transcription
        """
        session_id = str(uuid4())
        session = WhisperSession(
            session_id=session_id,
            model=self,
            language=language,
            sample_rate=sample_rate,
            verbose=verbose
        )
        
        if verbose:
            print(f"Created FasterWhisper transcription session: {session_id}")
        
        return session

    def transcribe_core(
        self, 
        *, 
        path: Optional[str] = None,
        url: Optional[str] = None,
        blob: Optional[str] = None,
        language: Optional[str] = None,
        diarize: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Generator[Segment, None, None]:
        """
        Transcribe using faster-whisper engine.
        """
        # Handle URL download or blob processing if needed
        audio_path = utils.get_audio_file_path(path=path, url=url, blob=blob, verbose=verbose)
        
        if verbose:
            print(f"Using faster-whisper engine with model: {self.model}")
            print(f"Processing file: {audio_path}")
            if self.model_object:
                print(f"Using pre-loaded model: {self.model_object}")
            if diarize:
                print("Diarization is enabled")
        
        if diarize:
            from .diarization import diarize as diarize_func, match_speaker_to_interval
            
            diarizition_df = diarize_func(
                audio=audio_path,
                device=self.device,
                checkpoint_path=kwargs.get("checkpoint_path", None),
                num_speakers=kwargs.get("num_speakers", None),
                min_speakers=kwargs.get("min_speakers", None),
                max_speakers=kwargs.get("max_speakers", None),
                use_auth_token=kwargs.get("use_auth_token", None),
                verbose=verbose,
            )
        try:
            # Transcribe using faster-whisper directly with file path
            segments, info = self.model_object.transcribe(audio_path, language=language, word_timestamps=True)
            
            # Yield each segment with proper structure
            for segment in segments:
                # Build extra_data dictionary
                extra_data = _copy_segment_extra_data(segment, language=language)
                
                # Process words if available
                words = []
                if hasattr(segment, 'words') and segment.words:
                    for word_data in segment.words:
                        word = Word(
                            word=word_data.word,
                            start=word_data.start,
                            end=word_data.end,
                            probability=getattr(word_data, 'probability', None)
                        )
                        words.append(word)
                
                # Create Segment object
                segment = Segment(
                    text=segment.text,
                    start=segment.start,
                    end=segment.end,
                    words=words,
                    extra_data=extra_data
                )

                if diarize:
                    speaker = match_speaker_to_interval(diarizition_df, start=segment.start, end=segment.end)
                    segment.speakers = [speaker]
                
                yield segment
                
        except Exception as e:
            if verbose:
                print(f"Error during transcription: {e}")
            raise
        
        finally:
            # Clean up temporary files created for URL downloads or blob processing
            if (url is not None or blob is not None) and os.path.exists(audio_path):
                os.remove(audio_path)
    



class StableWhisperModel(TranscriptionModel):
    """Stable Whisper transcription model"""
    
    def __init__(self, model: str, device: str = None, local_files_only: bool = False):
        super().__init__(engine="stable-whisper", model=model)
        
        self.model_path = model
        self.device = device if device else utils.guess_device()
        self.local_files_only = local_files_only
        
        # Load the model immediately
        self.model_object = self._load_stable_whisper_model()
    
    def _load_stable_whisper_model(self) -> Any:
        """
        Load the actual stable-whisper model.
        """
        # Import stable_whisper
        try:
            import stable_whisper
        except ImportError:
            raise ImportError("stable-whisper is not installed. Please install it with: pip install stable-whisper")
        
        device_index = None
        
        if len(self.device.split(",")) > 1:
            device_indexes = []
            base_device = None
            for device_instance in self.device.split(","):
                device, device_index = get_device_and_index(device_instance)
                base_device = base_device or device
                if base_device != device:
                    raise ValueError("Multiple devices must be instances of the same base device (e.g cuda:0, cuda:1 etc.)")
                device_indexes.append(device_index)
            device = base_device
            device_index = device_indexes
        else:
            device, device_index = get_device_and_index(self.device)
        
        args = {'device': device}
        if device_index:
            args['device_index'] = device_index
        if self.local_files_only:
            args['local_files_only'] = self.local_files_only
        
        print(f'Loading stable-whisper model: {self.model_path} on {device} with index: {device_index or 0}')
        return stable_whisper.load_faster_whisper(self.model_path, **args)
    
    def create_session(self, language: Optional[str] = None, sample_rate: int = 16000, 
                      verbose: bool = False) -> TranscriptionSession:
        """
        Create a new transcription session for incremental audio processing.
        
        Args:
            language: Language code for transcription (e.g., 'he' for Hebrew, 'en' for English)
            sample_rate: Audio sample rate (default: 16000 Hz)
            verbose: Whether to enable verbose output
            
        Returns:
            WhisperSession object for incremental transcription
        """
        session_id = str(uuid4())
        session = WhisperSession(
            session_id=session_id,
            model=self,
            language=language,
            sample_rate=sample_rate,
            verbose=verbose
        )
        
        if verbose:
            print(f"Created StableWhisper transcription session: {session_id}")
        
        return session

    def transcribe_core(
        self, 
        *, 
        path: Optional[str] = None,
        url: Optional[str] = None,
        blob: Optional[str] = None,
        language: Optional[str] = None,
        diarize: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Generator[Segment, None, None]:
        """
        Transcribe using stable-whisper engine.
        """
        # Handle URL download or blob processing if needed
        audio_path = utils.get_audio_file_path(path=path, url=url, blob=blob, verbose=verbose)
        
        if verbose:
            print(f"Using stable-whisper engine with model: {self.model}")
            print(f"Processing file: {audio_path}")
            if self.model_object:
                print(f"Using pre-loaded model: {self.model_object}")
            if diarize:
                print("Diarization is enabled")
        
        if diarize:
            from .diarization import diarize as diarize_func, match_speaker_to_interval
            
            diarizition_df = diarize_func(
                audio=audio_path,
                device=self.device,
                checkpoint_path=kwargs.get("checkpoint_path", None),
                num_speakers=kwargs.get("num_speakers", None),
                min_speakers=kwargs.get("min_speakers", None),
                max_speakers=kwargs.get("max_speakers", None),
                use_auth_token=kwargs.get("use_auth_token", None),
                verbose=verbose,
            )   
        try:
            # Transcribe using stable-whisper with word timestamps
            result = self.model_object.transcribe(audio_path, language=language, word_timestamps=True)
            segments = result.segments
            
            # Yield each segment with proper structure
            for segment in segments:
                # Build extra_data dictionary
                extra_data = _copy_segment_extra_data(segment, language=language)
                
                # Process words if available
                words = []
                if hasattr(segment, 'words') and segment.words:
                    for word_data in segment.words:
                        word = Word(
                            word=word_data.word,
                            start=word_data.start,
                            end=word_data.end,
                            probability=getattr(word_data, 'probability', None)
                        )
                        words.append(word)
                
                # Create Segment object
                segment = Segment(
                    text=segment.text,
                    start=segment.start,
                    end=segment.end,
                    words=words,
                    extra_data=extra_data
                )

                if diarize:
                    speaker = match_speaker_to_interval(diarizition_df, start=segment.start, end=segment.end)
                    segment.speakers = [speaker]
                
                yield segment
                
        except Exception as e:
            if verbose:
                print(f"Error during transcription: {e}")
            raise
        
        finally:
            # Clean up temporary files created for URL downloads or blob processing
            if (url is not None or blob is not None) and os.path.exists(audio_path):
                os.remove(audio_path)
    


class RunPodJob:
    def __init__(self, api_key: str, endpoint_id: str, payload: dict):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # Submit the job immediately on creation
        response = requests.post(
            f"{self.base_url}/run",
            headers=self.headers,
            json=payload
        )

        if response.status_code == 401:
            raise Exception("Invalid RunPod API key")

        response.raise_for_status()

        result = response.json()
        self.job_id = result.get("id")

    def status(self):
        """Get job status"""
        response = requests.get(
            f"{self.base_url}/status/{self.job_id}",
            headers=self.headers
        )
        response.raise_for_status()

        status_response = response.json()
        return status_response.get("status", "UNKNOWN")

    def stream(self):
        """Stream job results"""
        while True:
            response = requests.get(
                f"{self.base_url}/stream/{self.job_id}",
                headers=self.headers,
                stream=True
            )
            response.raise_for_status()

            # Expect a single response
            try:
                content = response.content.decode('utf-8')
                data = json.loads(content)
                if data['status'] not in ['IN_PROGRESS', 'COMPLETED']:
                    break

                for item in data['stream']:
                    # Decode JSON result
                    output = item['output']
                    try:
                        # Parse JSON and reconstruct Segment object
                        decoded_output = Segment(**output)
                        yield decoded_output
                    except Exception as e:
                        # If JSON decode fails, raise the exception
                        raise Exception(f"Failed to decode JSON: {e}")

                if data['status'] == 'COMPLETED':
                    return

            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {e}")
                return

    def cancel(self):
        """Cancel the job"""
        response = requests.post(
            f"{self.base_url}/cancel/{self.job_id}",
            headers=self.headers
        )
        response.raise_for_status()

        return response.json()


class AsyncRunPodJob:
    """Async version of RunPodJob"""
    
    def __init__(self, api_key: str, endpoint_id: str, payload: dict):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.payload = payload
        self.job_id = None

    async def submit(self):
        """Submit the job asynchronously"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/run",
                headers=self.headers,
                json=self.payload
            ) as response:
                if response.status == 401:
                    raise Exception("Invalid RunPod API key")
                
                response.raise_for_status()
                result = await response.json()
                self.job_id = result.get("id")

    async def status(self):
        """Get job status asynchronously"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/status/{self.job_id}",
                headers=self.headers
            ) as response:
                response.raise_for_status()
                status_response = await response.json()
                return status_response.get("status", "UNKNOWN")

    async def stream(self):
        """Stream job results asynchronously"""
        while True:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/stream/{self.job_id}",
                    headers=self.headers
                ) as response:
                    response.raise_for_status()
                    
                    # Expect a single response
                    try:
                        content = await response.text()
                        data = json.loads(content)
                        if data['status'] not in ['IN_PROGRESS', 'COMPLETED']:
                            break

                        for item in data['stream']:
                            # Decode JSON result
                            output = item['output']
                            try:
                                # Parse JSON and reconstruct Segment object
                                decoded_output = Segment(**output)
                                yield decoded_output
                            except Exception as e:
                                # If JSON decode fails, raise the exception
                                raise Exception(f"Failed to decode JSON: {e}")

                        if data['status'] == 'COMPLETED':
                            return

                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON response: {e}")
                        return

    async def cancel(self):
        """Cancel the job asynchronously"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/cancel/{self.job_id}",
                headers=self.headers
            ) as response:
                response.raise_for_status()
                return await response.json()


class RunPodModel(TranscriptionModel):
    """RunPod transcription model"""
    
    def __init__(self, model: str, api_key: str, endpoint_id: str, core_engine: str = "faster-whisper"):
        super().__init__(engine="runpod", model=model)
        
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        
        # Validate core engine
        if core_engine not in ["faster-whisper", "stable-whisper"]:
            raise ValueError(f"Unsupported core engine: {core_engine}. Supported engines: 'faster-whisper', 'stable-whisper'")
        
        self.core_engine = core_engine
        
        # Constants for RunPod
        self.IN_QUEUE_TIMEOUT = 300
        self.MAX_STREAM_TIMEOUTS = 5
        self.RUNPOD_MAX_PAYLOAD_LEN = 10 * 1024 * 1024
    
    def create_session(self, language: Optional[str] = None, sample_rate: int = 16000, 
                      verbose: bool = False) -> TranscriptionSession:
        """
        Create a new transcription session for incremental audio processing.
        
        Note: RunPod sessions have limited functionality since they rely on remote API calls.
        The session will accumulate audio but cannot perform true incremental transcription
        until flush() is called.
        
        Args:
            language: Language code for transcription (e.g., 'he' for Hebrew, 'en' for English)
            sample_rate: Audio sample rate (default: 16000 Hz)
            verbose: Whether to enable verbose output
            
        Returns:
            WhisperSession object for incremental transcription
        """
        session_id = str(uuid4())
        session = WhisperSession(
            session_id=session_id,
            model=self,
            language=language,
            sample_rate=sample_rate,
            verbose=verbose
        )
        
        if verbose:
            print(f"Created RunPod transcription session: {session_id}")
            print("Note: RunPod sessions buffer audio locally and transcribe on flush()")
        
        return session

    def transcribe_core(
        self, 
        *, 
        path: Optional[str] = None,
        url: Optional[str] = None,
        blob: Optional[str] = None,
        language: Optional[str] = None,
        diarize: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Generator[Segment, None, None]:
        """
        Transcribe using RunPod engine.
        """
        # Determine payload type and data
        if path is not None:
            payload_type = "blob"
            data_source = path
        elif url is not None:
            payload_type = "url"
            data_source = url
        elif blob is not None:
            payload_type = "blob"
            data_source = blob
        else:
            raise ValueError("Must specify either 'path', 'url', or 'blob'")
        
        if verbose:
            print(f"Using RunPod engine with model: {self.model}")
            print(f"Payload type: {payload_type}")
            print(f"Data source: {data_source}")
        
        # Prepare payload
        payload = {
            "input": {
                "type": payload_type,
                "model": self.model,
                "engine": self.core_engine,
                "streaming": True,
                "transcribe_args": {
                    "language": language,
                    "diarize": diarize,
                    "verbose": verbose,
                    **kwargs
                }
            }
        }
        
        if payload_type == "blob":
            if path is not None:
                # Read audio file and encode as base64
                try:
                    with open(data_source, 'rb') as f:
                        audio_data = f.read()
                    payload["input"]["transcribe_args"]["blob"] = base64.b64encode(audio_data).decode('utf-8')
                except Exception as e:
                    raise Exception(f"Failed to read audio file: {e}")
            else:
                # Use blob data directly
                payload["input"]["transcribe_args"]["blob"] = data_source
        else:
            payload["input"]["transcribe_args"]["url"] = data_source
        
        # Check payload size
        if len(str(payload)) > self.RUNPOD_MAX_PAYLOAD_LEN:
            raise ValueError(f"Payload length is {len(str(payload))}, exceeding max payload length of {self.RUNPOD_MAX_PAYLOAD_LEN}")
        
        # Create and execute RunPod job
        run_request = RunPodJob(self.api_key, self.endpoint_id, payload)
        
        # Wait for task to be queued
        if verbose:
            print("Waiting for task to be queued...")
        
        for i in range(self.IN_QUEUE_TIMEOUT):
            if run_request.status() == "IN_QUEUE":
                time.sleep(1)
                continue
            break
        
        if verbose:
            print(f"Task status: {run_request.status()}")
        
        # Collect streaming results
        timeouts = 0
        while True:
            try:
                for segment_data in run_request.stream():
                    if isinstance(segment_data, Segment):
                        yield segment_data
                    else:
                        raise Exception(f"RunPod error: {segment_data}")

                # If we get here, streaming is complete
                run_request = None
                break
                
            except requests.exceptions.ReadTimeout:
                timeouts += 1
                if timeouts > self.MAX_STREAM_TIMEOUTS:
                    raise Exception(f"Number of request.stream() timeouts exceeded the maximum ({self.MAX_STREAM_TIMEOUTS})")
                if verbose:
                    print(f"Stream timeout {timeouts}/{self.MAX_STREAM_TIMEOUTS}, retrying...")
                continue
                
            except Exception as e:
                run_request.cancel()
                run_request = None
                raise Exception(f"Exception during RunPod streaming: {e}")

            finally:
                if run_request:
                    run_request.cancel()

    async def transcribe_async(
        self,
        *,
        path: Optional[str] = None,
        url: Optional[str] = None,
        blob: Optional[str] = None,
        language: Optional[str] = None,
        diarize: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> AsyncGenerator[Segment, None]:
        """
        Transcribe audio using this model asynchronously.
        
        Args:
            path: Path to the audio file to transcribe (mutually exclusive with url and blob)
            url: URL to download and transcribe (mutually exclusive with path and blob)
            blob: Base64 encoded blob data to transcribe (mutually exclusive with path and url)
            language: Language code for transcription (e.g., 'he' for Hebrew, 'en' for English)
            diarize: Whether to enable speaker diarization  
            verbose: Whether to enable verbose output
            **kwargs: Additional keyword arguments for the transcription model.
        Returns:
            AsyncGenerator yielding transcription segments
            
        Raises:
            ValueError: If multiple input sources are provided, or none is provided
            FileNotFoundError: If the specified path doesn't exist
            Exception: For other transcription errors
        """
        # Validate arguments
        provided_args = [arg for arg in [path, url, blob] if arg is not None]
        if len(provided_args) > 1:
            raise ValueError("Cannot specify multiple input sources - path, url, and blob are mutually exclusive")
        
        if len(provided_args) == 0:
            raise ValueError("Must specify either 'path', 'url', or 'blob'")

        # Determine payload type and data
        if path is not None:
            payload_type = "blob"
            data_source = path
        elif url is not None:
            payload_type = "url"
            data_source = url
        elif blob is not None:
            payload_type = "blob"
            data_source = blob
        else:
            raise ValueError("Must specify either 'path', 'url', or 'blob'")
        
        if diarize:
            raise NotImplementedError("Diarization is not supported for RunPod engine")
        
        if verbose:
            print(f"Using RunPod engine with model: {self.model}")
            print(f"Payload type: {payload_type}")
            print(f"Data source: {data_source}")
        
        # Prepare payload
        payload = {
            "input": {
                "type": payload_type,
                "model": self.model,
                "engine": self.core_engine,
                "streaming": True,
                "transcribe_args": {
                    "language": language,
                    "diarize": diarize,
                    "verbose": verbose,
                    **kwargs
                }
            }
        }
        
        if payload_type == "blob":
            if path is not None:
                # Read audio file and encode as base64
                try:
                    with open(data_source, 'rb') as f:
                        audio_data = f.read()
                    payload["input"]["transcribe_args"]["blob"] = base64.b64encode(audio_data).decode('utf-8')
                except Exception as e:
                    raise Exception(f"Failed to read audio file: {e}")
            else:
                # Use blob data directly
                payload["input"]["transcribe_args"]["blob"] = data_source
        else:
            payload["input"]["transcribe_args"]["url"] = data_source
        
        # Check payload size
        if len(str(payload)) > self.RUNPOD_MAX_PAYLOAD_LEN:
            raise ValueError(f"Payload length is {len(str(payload))}, exceeding max payload length of {self.RUNPOD_MAX_PAYLOAD_LEN}")
        
        # Create and execute RunPod job
        run_request = AsyncRunPodJob(self.api_key, self.endpoint_id, payload)
        
        # Submit the job
        await run_request.submit()
        
        # Wait for task to be queued
        if verbose:
            print("Waiting for task to be queued...")
        
        for i in range(self.IN_QUEUE_TIMEOUT):
            status = await run_request.status()
            if status == "IN_QUEUE":
                await asyncio.sleep(1)
                continue
            break
        
        if verbose:
            print(f"Task status: {await run_request.status()}")
        
        # Collect streaming results
        timeouts = 0
        while True:
            try:
                async for segment_data in run_request.stream():
                    if isinstance(segment_data, Segment):
                        yield segment_data
                    else:
                        raise Exception(f"RunPod error: {segment_data}")

                # If we get here, streaming is complete
                run_request = None
                break 
            except aiohttp.ClientError as e:
                timeouts += 1
                if timeouts > self.MAX_STREAM_TIMEOUTS:
                    raise Exception(f"Number of request.stream() timeouts exceeded the maximum ({self.MAX_STREAM_TIMEOUTS})")
                if verbose:
                    print(f"Stream timeout {timeouts}/{self.MAX_STREAM_TIMEOUTS}, retrying...")
                continue
                
            except Exception as e:
                await run_request.cancel()
                run_request = None
                raise Exception(f"Exception during RunPod streaming: {e}")

            finally:
                if run_request:
                    await run_request.cancel()
    


def load_model(
    *,
    engine: str,
    model: str,
    **kwargs
) -> TranscriptionModel:
    """
    Load a transcription model for the specified engine and model.
    
    Args:
        engine: Transcription engine to use ('faster-whisper', 'stable-whisper', 'runpod', or 'stable-ts')
        model: Model name for the selected engine
        **kwargs: Additional arguments for specific engines:
            - faster-whisper: device, local_files_only
            - stable-whisper: device, local_files_only
            - runpod: api_key (required), endpoint_id (required), core_engine
            - stable-ts: (future implementation)
        
    Returns:
        TranscriptionModel object that can be used for transcription
        
    Raises:
        ValueError: If the engine is not supported or required parameters are missing
        ImportError: If required dependencies are not installed
    """
    if engine == "faster-whisper":
        return FasterWhisperModel(model=model, **kwargs)
    elif engine == "stable-whisper":
        return StableWhisperModel(model=model, **kwargs)
    elif engine == "runpod":
        return RunPodModel(model=model, **kwargs)
    elif engine == "stable-ts":
        # Placeholder for future implementation
        raise NotImplementedError("stable-ts engine not yet implemented")
    else:
        raise ValueError(f"Unsupported engine: {engine}. Supported engines: 'faster-whisper', 'stable-whisper', 'runpod', 'stable-ts'")
