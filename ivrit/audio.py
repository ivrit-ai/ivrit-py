"""
Audio transcription functionality for ivrit.ai
"""
import base64
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Generator, Optional, Union

import requests

from .types import Segment
from .utils import get_audio_file_path


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
        diarize: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Union[dict, Generator]:
        """
        Transcribe audio using this model.
        
        Args:
            path: Path to the audio file to transcribe (mutually exclusive with url)
            url: URL to download and transcribe (mutually exclusive with path)
            language: Language code for transcription (e.g., 'he' for Hebrew, 'en' for English)
            stream: Whether to return results as a generator (True) or full result (False)
            diarize: Whether to enable speaker diarization  
            verbose: Whether to enable verbose output
            **kwargs: Additional keyword arguments for the transcription model.
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
        segments_generator = self.transcribe_core(path=path, url=url, language=language, diarize=diarize, verbose=verbose, **kwargs)
        
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
        language: Optional[str] = None,
        diarize: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Generator[Segment, None, None]:
        """
        Core transcription method that must be implemented by derived classes.
        
        Args:
            path: Path to the audio file to transcribe (mutually exclusive with url)
            url: URL to download and transcribe (mutually exclusive with path)
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
        diarize: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Generator[Segment, None, None]:
        """
        Transcribe using faster-whisper engine.
        """
        # Handle URL download if needed
        audio_path = get_audio_file_path(path=path, url=url, verbose=verbose)
        
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
                segment = Segment(
                    text=segment.text,
                    start=segment.start,
                    end=segment.end,
                    extra_data=extra_data
                )

                if diarize:
                    speaker = match_speaker_to_interval(diarizition_df, start=segment.start, end=segment.end)
                    segment.speaker = speaker
                
                yield segment
                
        except Exception as e:
            if verbose:
                print(f"Error during transcription: {e}")
            raise
        
        finally:
            if url is not None and os.path.exists(audio_path):
                os.remove(audio_path)


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
        diarize: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Generator[Segment, None, None]:
        """
        Transcribe using stable-whisper engine.
        """
        # Handle URL download if needed
        audio_path = get_audio_file_path(path=path, url=url, verbose=verbose)
        
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
                segment = Segment(
                    text=segment.text,
                    start=segment.start,
                    end=segment.end,
                    extra_data=extra_data
                )

                if diarize:
                    speaker = match_speaker_to_interval(diarizition_df, start=segment.start, end=segment.end)
                    segment.speaker = speaker
                
                yield segment
                
        except Exception as e:
            if verbose:
                print(f"Error during transcription: {e}")
            raise
        
        finally:
            if url is not None and os.path.exists(audio_path):
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
                    yield item['output']

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
    
    def transcribe_core(
        self, 
        *, 
        path: Optional[str] = None,
        url: Optional[str] = None,
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
        else:
            raise ValueError("Must specify either 'path' or 'url'")
        
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
                
            except requests.exceptions.ReadTimeout:
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
        engine: Transcription engine to use ('faster-whisper', 'stable-whisper', 'runpod', or 'stable-ts')
        model: Model name for the selected engine
        **kwargs: Additional arguments for specific engines:
            - faster-whisper: device
            - stable-whisper: device
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



