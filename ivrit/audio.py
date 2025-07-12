"""
Audio transcription functionality for ivrit.ai
"""

import os
import tempfile
import urllib.request
from typing import Generator, Union, Optional, Any, Dict
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
import base64
import time
import requests


@dataclass
class Segment:
    """Represents a transcription segment"""
    text: str
    start: float
    end: float
    extra_data: Dict[str, Any]


class TranscriptionModel(ABC):
    """Base class for transcription models"""
    
    def __init__(self, engine: str, model: str, model_object: Any = None):
        self.engine = engine
        self.model = model
        self.model_object = model_object
    
    def __repr__(self):
        return f"{self.__class__.__name__}(engine='{self.engine}', model='{self.model}')"
    
    def transcribe(
        self,
        *,
        path: Optional[str] = None,
        url: Optional[str] = None,
        language: Optional[str] = None,
        stream: bool = False,
        verbose: bool = False
    ) -> Union[dict, Generator]:
        """
        Transcribe audio using this model.
        
        Args:
            path: Path to the audio file to transcribe (mutually exclusive with url)
            url: URL to download and transcribe (mutually exclusive with path)
            language: Language code for transcription (e.g., 'he' for Hebrew, 'en' for English)
            stream: Whether to return results as a generator (True) or full result (False)
            verbose: Whether to enable verbose output
            
        Returns:
            If stream=True: Generator yielding transcription segments
            If stream=False: Complete transcription result as dictionary
            
        Raises:
            ValueError: If both path and url are provided, or neither is provided
            FileNotFoundError: If the specified path doesn't exist
            Exception: For other transcription errors
        """
        # Validate arguments
        if path is not None and url is not None:
            raise ValueError("Cannot specify both 'path' and 'url' - they are mutually exclusive")
        
        if path is None and url is None:
            raise ValueError("Must specify either 'path' or 'url'")
        
        # Get streaming results from the model
        segments_generator = self.transcribe_core(path=path, url=url, language=language, verbose=verbose)
        
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
            
            return {
                "text": full_text,
                "segments": segments,
                "language": segments[0].extra_data.get("language", language or "unknown"),
                "engine": self.engine,
                "model": self.model
            }
    
    @abstractmethod
    def transcribe_core(
        self, 
        *, 
        path: Optional[str] = None,
        url: Optional[str] = None,
        language: Optional[str] = None,
        verbose: bool = False
    ) -> Generator[Segment, None, None]:
        """
        Core transcription method that must be implemented by derived classes.
        
        Args:
            path: Path to the audio file to transcribe (mutually exclusive with url)
            url: URL to download and transcribe (mutually exclusive with path)
            language: Language code for transcription
            verbose: Whether to enable verbose output
            
        Returns:
            Generator yielding Segment objects
        """
        pass


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


class FasterWhisperModel(TranscriptionModel):
    """Faster Whisper transcription model"""
    
    def __init__(self, model: str, device: str = "auto"):
        super().__init__(engine="faster-whisper", model=model)
        
        self.model_path = model
        self.device = device
        
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
        
        print(f'Loading faster-whisper model: {self.model_path} on {device} with index: {device_index or 0}')
        return faster_whisper.WhisperModel(self.model_path, **args)
    
    def transcribe_core(
        self, 
        *, 
        path: Optional[str] = None,
        url: Optional[str] = None,
        language: Optional[str] = None,
        verbose: bool = False
    ) -> Generator[Segment, None, None]:
        """
        Transcribe using faster-whisper engine.
        """
        # Handle URL download if needed
        audio_path = path
        temp_file = None
        
        if url is not None:
            if verbose:
                print(f"Downloading audio from: {url}")
            
            temp_file = tempfile.NamedTemporaryFile(suffix=".audio")
            urllib.request.urlretrieve(url, temp_file.name)
            audio_path = temp_file.name
        
        # Validate file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if verbose:
            print(f"Using faster-whisper engine with model: {self.model}")
            print(f"Processing file: {audio_path}")
            if self.model_object:
                print(f"Using pre-loaded model: {self.model_object}")
        
        try:
            # Transcribe using faster-whisper directly with file path
            segments, info = self.model_object.transcribe(audio_path, language=language)
            
            # Yield each segment with proper structure
            for segment in segments:
                # Build extra_data dictionary
                extra_data = {
                    "info": {
                        "language": info.language,
                        "language_probability": info.language_probability
                    }
                }
                
                # Add all segment attributes to extra_data
                for attr_name in dir(segment):
                    if not attr_name.startswith('_') and attr_name not in ['text', 'start', 'end']:
                        try:
                            attr_value = getattr(segment, attr_name)
                            extra_data[attr_name] = attr_value
                        except Exception:
                            # Skip attributes that can't be accessed
                            pass
                
                # Create Segment object
                yield Segment(
                    text=segment.text,
                    start=segment.start,
                    end=segment.end,
                    extra_data=extra_data
                )
                
        except Exception as e:
            if verbose:
                print(f"Error during transcription: {e}")
            raise


class StableWhisperModel(TranscriptionModel):
    """Stable Whisper transcription model"""
    
    def __init__(self, model: str, device: str = "auto"):
        super().__init__(engine="stable-whisper", model=model)
        
        self.model_path = model
        self.device = device
        
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
        
        print(f'Loading stable-whisper model: {self.model_path} on {device} with index: {device_index or 0}')
        return stable_whisper.load_faster_whisper(self.model_path, **args)
    
    def transcribe_core(
        self, 
        *, 
        path: Optional[str] = None,
        url: Optional[str] = None,
        language: Optional[str] = None,
        verbose: bool = False
    ) -> Generator[Segment, None, None]:
        """
        Transcribe using stable-whisper engine.
        """
        # Handle URL download if needed
        audio_path = path
        temp_file = None
        
        if url is not None:
            if verbose:
                print(f"Downloading audio from: {url}")
            
            temp_file = tempfile.NamedTemporaryFile(suffix=".audio")
            urllib.request.urlretrieve(url, temp_file.name)
            audio_path = temp_file.name
        
        # Validate file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if verbose:
            print(f"Using stable-whisper engine with model: {self.model}")
            print(f"Processing file: {audio_path}")
            if self.model_object:
                print(f"Using pre-loaded model: {self.model_object}")
        
        try:
            # Transcribe using stable-whisper with word timestamps
            result = self.model_object.transcribe(audio_path, language=language, word_timestamps=True)
            segments = result.segments
            
            # Yield each segment with proper structure
            for segment in segments:
                # Build extra_data dictionary
                extra_data = {
                    "language": language or "unknown"
                }
                
                # Add all segment attributes to extra_data
                for attr_name in dir(segment):
                    if not attr_name.startswith('_') and attr_name not in ['text', 'start', 'end']:
                        try:
                            attr_value = getattr(segment, attr_name)
                            extra_data[attr_name] = attr_value
                        except Exception:
                            # Skip attributes that can't be accessed
                            pass
                
                # Create Segment object
                yield Segment(
                    text=segment.text,
                    start=segment.start,
                    end=segment.end,
                    extra_data=extra_data
                )
                
        except Exception as e:
            if verbose:
                print(f"Error during transcription: {e}")
            raise


class RunPodModel(TranscriptionModel):
    """RunPod transcription model"""
    
    def __init__(self, model: str, api_key: Optional[str] = None, endpoint_id: Optional[str] = None, core_engine: str = "faster-whisper"):
        super().__init__(engine="runpod", model=model)
        
        # Get API key and endpoint ID from environment or parameters
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self.endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID")
        
        if not self.api_key:
            raise ValueError("RunPod API key must be provided via 'api_key' parameter or RUNPOD_API_KEY environment variable")
        
        if not self.endpoint_id:
            raise ValueError("RunPod endpoint ID must be provided via 'endpoint_id' parameter or RUNPOD_ENDPOINT_ID environment variable")
        
        # Validate core engine
        if core_engine not in ["faster-whisper", "stable-whisper"]:
            raise ValueError(f"Unsupported core engine: {core_engine}. Supported engines: 'faster-whisper', 'stable-whisper'")
        
        self.core_engine = core_engine
        
        # Constants for RunPod
        self.IN_QUEUE_TIMEOUT = 300
        self.MAX_STREAM_TIMEOUTS = 5
        self.RUNPOD_MAX_PAYLOAD_LEN = 10 * 1024 * 1024
        
        # Load runpod package
        try:
            import runpod
            self.runpod = runpod
        except ImportError:
            raise ImportError("runpod is not installed. Please install it with: pip install runpod")
    
    def transcribe_core(
        self, 
        *, 
        path: Optional[str] = None,
        url: Optional[str] = None,
        language: Optional[str] = None,
        verbose: bool = False
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
        else:
            raise ValueError("Must specify either 'path' or 'url'")
        
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
                "streaming": True
            }
        }
        
        if payload_type == "blob":
            # Read audio file and encode as base64
            try:
                with open(data_source, 'rb') as f:
                    audio_data = f.read()
                payload["input"]["data"] = base64.b64encode(audio_data).decode('utf-8')
            except Exception as e:
                raise Exception(f"Failed to read audio file: {e}")
        else:
            payload["input"]["url"] = data_source
        
        # Check payload size
        if len(str(payload)) > self.RUNPOD_MAX_PAYLOAD_LEN:
            raise ValueError(f"Payload length is {len(str(payload))}, exceeding max payload length of {self.RUNPOD_MAX_PAYLOAD_LEN}")
        
        # Configure runpod endpoint and execute
        self.runpod.api_key = self.api_key
        ep = self.runpod.Endpoint(self.endpoint_id)
        run_request = ep.run(payload)
        
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
                    if "error" in segment_data:
                        raise Exception(f"RunPod error: {segment_data['error']}")
                    
                    # Extract segment information from well-formatted RunPod data
                    text = segment_data["text"]
                    start = segment_data["start"]
                    end = segment_data["end"]
                    
                    # Build extra_data dictionary
                    extra_data = {
                        "runpod_segment": segment_data,
                        "language": language or "unknown"
                    }
                    
                    # Add any additional fields from the segment
                    for key, value in segment_data.items():
                        if key not in ["text", "start", "end"]:
                            extra_data[key] = value
                    
                    yield Segment(
                        text=text,
                        start=start,
                        end=end,
                        extra_data=extra_data
                    )
                
                # If we get here, streaming is complete
                break
                
            except requests.exceptions.ReadTimeout as e:
                timeouts += 1
                if timeouts > self.MAX_STREAM_TIMEOUTS:
                    raise Exception(f"Number of request.stream() timeouts exceeded the maximum ({self.MAX_STREAM_TIMEOUTS})")
                if verbose:
                    print(f"Stream timeout {timeouts}/{self.MAX_STREAM_TIMEOUTS}, retrying...")
                continue
                
            except Exception as e:
                run_request.cancel()
                raise Exception(f"Exception during RunPod streaming: {e}")


def load_model(
    *,
    engine: str,
    model: str,
    **kwargs
) -> TranscriptionModel:
    """
    Load a transcription model for the specified engine and model.
    
    Args:
        engine: Transcription engine to use ('faster-whisper', 'stable-ts', or 'runpod')
        model: Model name for the selected engine
        **kwargs: Additional arguments for specific engines:
            - faster-whisper: device
            - stable-whisper: device
            - runpod: api_key, endpoint_id, core_engine
            - stable-ts: (future implementation)
        
    Returns:
        TranscriptionModel object that can be used for transcription
        
    Raises:
        ValueError: If the engine is not supported
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



