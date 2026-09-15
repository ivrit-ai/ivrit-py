"""
Audio transcription functionality for ivrit.ai
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
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
from .utils import ProgressCallback, emit_progress, invoke_progress

logger = logging.getLogger(__name__)


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

    def _normalize_sources(
        self,
        path: Optional[Union[str, List[str]]],
        url: Optional[Union[str, List[str]]],
        blob: Optional[Union[str, List[str]]],
    ) -> tuple[str, List[str], bool]:
        """
        Validate and normalize the audio source arguments.

        Exactly one of ``path`` / ``url`` / ``blob`` must be provided. Each may
        be either a single string (single-file mode) or a list of strings
        (batch mode). Centralizes the mutual-exclusivity and empty-list checks.

        Args:
            path: Path source (string or list of strings)
            url: URL source (string or list of strings)
            blob: Base64 blob source (string or list of strings)

        Returns:
            Tuple of (kind, items, is_batch) where kind is "path"|"url"|"blob",
            items is the list of string sources, and is_batch is whether a list
            was passed.

        Raises:
            ValueError: If multiple source kinds are provided, none is provided,
                or a provided list is empty.
        """
        provided = [(kind, value) for kind, value in (("path", path), ("url", url), ("blob", blob)) if value is not None]
        if len(provided) > 1:
            raise ValueError("Cannot specify multiple input sources - path, url, and blob are mutually exclusive")
        if len(provided) == 0:
            raise ValueError("Must specify either 'path', 'url', or 'blob'")

        kind, value = provided[0]
        if isinstance(value, list):
            if len(value) == 0:
                raise ValueError(f"Empty source list provided for '{kind}'")
            return kind, value, True
        return kind, [value], False

    def _build_result_dict(self, segments: List[Segment], language: Optional[str]) -> dict:
        """
        Build the single-file transcription result dictionary from segments.

        Reproduces the empty-segments branch and the language resolution from
        ``segments[0].extra_data`` exactly as the single-file path requires.

        Args:
            segments: List of transcription segments
            language: Language code requested by the caller, if any

        Returns:
            Transcription result dictionary
        """
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

    def _wrap_progress(
        self,
        on_progress: Optional[ProgressCallback],
        index: int,
        total: int,
    ) -> Optional[ProgressCallback]:
        """
        Wrap a progress callback to inject batch attribution into each event.

        Merges ``batch_index`` / ``batch_total`` into the event's ``extra`` dict
        without clobbering engine-supplied extras. Returns None when the user
        did not supply a callback.

        Args:
            on_progress: The user-supplied progress callback, or None
            index: The position of this item in the batch
            total: The number of items in the batch

        Returns:
            A wrapped callback, or None if on_progress is None
        """
        if on_progress is None:
            return None

        def wrapped(event: Dict[str, Any]) -> None:
            event["extra"] = {**(event.get("extra", {}) or {}), "batch_index": index, "batch_total": total}
            on_progress(event)

        return wrapped

    def transcribe(
        self,
        *,
        path: Optional[Union[str, List[str]]] = None,
        url: Optional[Union[str, List[str]]] = None,
        blob: Optional[Union[str, List[str]]] = None,
        language: Optional[str] = None,
        stream: bool = False,
        diarize: bool = False,
        diarization_args: Optional[Dict[str, Any]] = None,
        output_options: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        on_progress: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> Union[dict, Generator, List[dict]]:
        """
        Transcribe audio using this model.

        Each of path/url/blob accepts either a single string (single-file mode)
        or a list of strings (batch mode). A list - even a one-element list -
        selects batch mode and drives the return type. Batch items are processed
        strictly sequentially with per-item error isolation.

        Args:
            path: Path(s) to the audio file to transcribe; string or list of strings
                (mutually exclusive with url and blob)
            url: URL(s) to download and transcribe; string or list of strings
                (mutually exclusive with path and blob)
            blob: Base64 encoded blob data to transcribe; string or list of strings
                (mutually exclusive with path and url)
            language: Language code for transcription (e.g., 'he' for Hebrew, 'en' for English)
            stream: Whether to return results as a generator (True) or full result (False)
            diarize: Whether to enable speaker diarization
            diarization_args: Dictionary of arguments for diarization (engine, device, num_speakers, etc.)
            output_options: Dictionary controlling output verbosity. Supported keys:
                - word_timestamps (bool): Whether to populate word-level timestamps (default: True)
                - extra_data (bool): Whether to populate extra metadata fields (default: True)
            verbose: Whether to enable verbose output
            on_progress: Optional callback invoked periodically as work
                advances. Receives a dict with the following core keys:
                  - 'phase': str, either 'transcription' or 'diarization'.
                    A single callback may be invoked with both values when a
                    transcribe-then-diarize run is in progress; events are
                    emitted in phase order.
                  - 'step': str — a sub-phase label such as 'decode',
                    'embedding', or 'clustering'.
                  - 'step_fraction': float — 0.0 to 1.0, progress within the
                    current step. 0.0 when the engine cannot compute it.
                  - 'description': str — short human-readable label suitable
                    for a UI progress indicator.
                  - 'extra': dict — engine-specific data (e.g.
                    'processed_seconds', 'total_seconds', 'segment_index',
                    'clusters_tried'). May be empty.
                Exceptions raised by the callback are caught and logged at
                warning level.
            **kwargs: Additional keyword arguments for the transcription model.
        Returns:
            Single-file mode (bare string source):
                If stream=True: Generator yielding transcription segments
                If stream=False: Complete transcription result as dictionary
            Batch mode (list source):
                If stream=True: Generator yielding (index, Segment) tuples, or
                    (index, Exception) on per-item failure
                If stream=False: List[dict], one entry per input in input order;
                    a failed item is {"error": str, "source": kind, "input": value}

        Raises:
            ValueError: If multiple input sources are provided, none is provided,
                an empty list is provided, or stream=True with diarize=True
            FileNotFoundError: If the specified path doesn't exist
            Exception: For other transcription errors
        """
        # Validate sources eagerly (before returning any generator) so misuse
        # errors propagate from the call itself.
        kind, items, is_batch = self._normalize_sources(path, url, blob)

        # Validate streaming with diarization eagerly. This is a misuse error,
        # not a per-item data error, so it must surface immediately.
        if stream and diarize:
            raise ValueError("Streaming (stream=True) is not compatible with diarization (diarize=True). Diarization requires processing all segments before speaker assignment.")

        if not is_batch:
            return self._transcribe_one(
                **{kind: items[0]},
                language=language,
                stream=stream,
                diarize=diarize,
                diarization_args=diarization_args,
                output_options=output_options,
                verbose=verbose,
                on_progress=on_progress,
                **kwargs,
            )

        return self._transcribe_batch(
            kind=kind,
            items=items,
            language=language,
            stream=stream,
            diarize=diarize,
            diarization_args=diarization_args,
            output_options=output_options,
            verbose=verbose,
            on_progress=on_progress,
            **kwargs,
        )

    def _transcribe_batch(
        self,
        *,
        kind: str,
        items: List[str],
        language: Optional[str] = None,
        stream: bool = False,
        diarize: bool = False,
        diarization_args: Optional[Dict[str, Any]] = None,
        output_options: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        on_progress: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> Union[Generator, List[dict]]:
        """
        Overridable batch seam for synchronous transcription.

        The default implementation fans each item out through the single-source
        ``_transcribe_one`` path in input order, with per-item error isolation
        and per-item progress attribution via ``_wrap_progress``. Subclasses
        (e.g. RunPod) override this to submit a single batched job instead.

        Source normalization/validation is owned by ``transcribe``; this method
        receives already-normalized ``kind`` / ``items``.

        Returns a ``Generator[Tuple[int, Union[Segment, Exception]]]`` when
        ``stream`` is True, else a ``List[dict]`` in input order.
        """
        total = len(items)

        if stream:
            def batch_generator():
                for i, item in enumerate(items):
                    try:
                        generator = self._transcribe_one(
                            **{kind: item},
                            language=language,
                            stream=True,
                            diarize=diarize,
                            diarization_args=diarization_args,
                            output_options=output_options,
                            verbose=verbose,
                            on_progress=self._wrap_progress(on_progress, i, total),
                            **kwargs,
                        )
                        for segment in generator:
                            yield (i, segment)
                    except Exception as exc:
                        yield (i, exc)
            return batch_generator()

        results: List[dict] = []
        for i, item in enumerate(items):
            try:
                results.append(self._transcribe_one(
                    **{kind: item},
                    language=language,
                    stream=False,
                    diarize=diarize,
                    diarization_args=diarization_args,
                    output_options=output_options,
                    verbose=verbose,
                    on_progress=self._wrap_progress(on_progress, i, total),
                    **kwargs,
                ))
            except Exception as exc:
                results.append({"error": str(exc), "source": kind, "input": item})
        return results

    async def _transcribe_batch_async(
        self,
        *,
        kind: str,
        items: List[str],
        language: Optional[str] = None,
        diarize: bool = False,
        diarization_args: Optional[Dict[str, Any]] = None,
        output_options: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        on_progress: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> AsyncGenerator[tuple[int, Union[Segment, Exception]], None]:
        """
        Overridable batch seam for asynchronous transcription (always streams).

        The default implementation fans each item out through the single-source
        ``_transcribe_one_async`` path in input order, with per-item error
        isolation and per-item progress attribution. Subclasses (e.g. RunPod)
        override this to submit a single batched job instead.

        Source normalization/validation is owned by ``transcribe_async``; this
        method receives already-normalized ``kind`` / ``items``.
        """
        total = len(items)
        for i, item in enumerate(items):
            try:
                async for segment in self._transcribe_one_async(
                    **{kind: item},
                    language=language,
                    diarize=diarize,
                    diarization_args=diarization_args,
                    output_options=output_options,
                    verbose=verbose,
                    on_progress=self._wrap_progress(on_progress, i, total),
                    **kwargs,
                ):
                    yield (i, segment)
            except Exception as exc:
                yield (i, exc)

    def _transcribe_one(
        self,
        *,
        path: Optional[str] = None,
        url: Optional[str] = None,
        blob: Optional[str] = None,
        language: Optional[str] = None,
        stream: bool = False,
        diarize: bool = False,
        diarization_args: Optional[Dict[str, Any]] = None,
        output_options: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        on_progress: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> Union[dict, Generator]:
        """
        Transcribe a single audio source. This is the single-file core path used
        by both single and batch calls. See ``transcribe`` for argument details.
        """
        # Default output options if not provided
        if output_options is None:
            output_options = {}

        # Set defaults for output options
        output_options = {
            'word_timestamps': output_options.get('word_timestamps', True),
            'extra_data': output_options.get('extra_data', True),
        }

        # Get streaming results from the model
        segments_generator = self.transcribe_core(path=path, url=url, blob=blob, language=language, diarize=diarize, diarization_args=diarization_args, output_options=output_options, verbose=verbose, on_progress=on_progress, **kwargs)

        if stream:
            # Return generator directly
            return segments_generator
        else:
            # Collect all segments and return as dictionary
            segments = list(segments_generator)
            return self._build_result_dict(segments, language)

    @abstractmethod
    def transcribe_core(
        self,
        *,
        path: Optional[str] = None,
        url: Optional[str] = None,
        blob: Optional[str] = None,
        language: Optional[str] = None,
        diarize: bool = False,
        diarization_args: Optional[Dict[str, Any]] = None,
        output_options: Dict[str, Any],
        verbose: bool = False,
        on_progress: Optional[ProgressCallback] = None,
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
            diarization_args: Dictionary of arguments for diarization (engine, device, num_speakers, etc.)
            output_options: Dictionary controlling output verbosity (word_timestamps, extra_data)
            verbose: Whether to enable verbose output
            on_progress: Optional progress callback. See TranscriptionModel.transcribe.
            **kwargs: Additional keyword arguments for the transcription model.

        Returns:
            Generator yielding Segment objects
        """

    async def transcribe_async(
        self,
        *,
        path: Optional[Union[str, List[str]]] = None,
        url: Optional[Union[str, List[str]]] = None,
        blob: Optional[Union[str, List[str]]] = None,
        language: Optional[str] = None,
        diarize: bool = False,
        diarization_args: Optional[Dict[str, Any]] = None,
        output_options: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        on_progress: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> AsyncGenerator[tuple[int, Union[Segment, Exception]], None]:
        """
        Transcribe audio using this model asynchronously.

        Each of path/url/blob accepts either a single string (single-file mode)
        or a list of strings (batch mode). Async transcription always streams.

        Single-file mode yields Segment objects. Batch mode yields
        (index, Segment) tuples, or a single (index, Exception) on per-item
        failure before continuing to the next item. Batch items are processed
        strictly sequentially in input order.

        Args:
            path: Path(s) to the audio file to transcribe; string or list of strings
                (mutually exclusive with url and blob)
            url: URL(s) to download and transcribe; string or list of strings
                (mutually exclusive with path and blob)
            blob: Base64 encoded blob data to transcribe; string or list of strings
                (mutually exclusive with path and url)
            language: Language code for transcription (e.g., 'he' for Hebrew, 'en' for English)
            diarize: Whether to enable speaker diarization
            diarization_args: Dictionary of arguments for diarization (engine, device, num_speakers, etc.)
            output_options: Dictionary controlling output verbosity. Supported keys:
                - word_timestamps (bool): Whether to populate word-level timestamps (default: True)
                - extra_data (bool): Whether to populate extra metadata fields (default: True)
            verbose: Whether to enable verbose output
            on_progress: Optional progress callback. See TranscriptionModel.transcribe.
                When the user-supplied callback may be invoked from a worker
                thread (default async impl), it must be thread-safe.
            **kwargs: Additional keyword arguments for the transcription model.
        Returns:
            Single-file mode: AsyncGenerator yielding Segment objects
            Batch mode: AsyncGenerator yielding (index, Segment) / (index, Exception)

        Raises:
            ValueError: If multiple input sources are provided, none is provided,
                or an empty list is provided
            FileNotFoundError: If the specified path doesn't exist
            Exception: For other transcription errors
        """
        # Validate sources eagerly (before entering the generator body) so
        # misuse errors propagate from the call itself.
        kind, items, is_batch = self._normalize_sources(path, url, blob)

        if not is_batch:
            async for segment in self._transcribe_one_async(
                **{kind: items[0]},
                language=language,
                diarize=diarize,
                diarization_args=diarization_args,
                output_options=output_options,
                verbose=verbose,
                on_progress=on_progress,
                **kwargs,
            ):
                yield segment
            return

        async for x in self._transcribe_batch_async(
            kind=kind,
            items=items,
            language=language,
            diarize=diarize,
            diarization_args=diarization_args,
            output_options=output_options,
            verbose=verbose,
            on_progress=on_progress,
            **kwargs,
        ):
            yield x

    async def _transcribe_one_async(
        self,
        *,
        path: Optional[str] = None,
        url: Optional[str] = None,
        blob: Optional[str] = None,
        language: Optional[str] = None,
        diarize: bool = False,
        diarization_args: Optional[Dict[str, Any]] = None,
        output_options: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        on_progress: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> AsyncGenerator[Segment, None]:
        """
        Transcribe a single audio source asynchronously.

        Runs the transcription in a thread pool to allow other coroutines to
        continue. This is the single-source async seam; subclasses (e.g. RunPod)
        may override it with a native async implementation. See
        ``transcribe_async`` for argument details.
        """
        # Validate arguments
        provided_args = [arg for arg in [path, url, blob] if arg is not None]
        if len(provided_args) > 1:
            raise ValueError("Cannot specify multiple input sources - path, url, and blob are mutually exclusive")

        if len(provided_args) == 0:
            raise ValueError("Must specify either 'path', 'url', or 'blob'")

        # Default output options if not provided
        if output_options is None:
            output_options = {}

        # Set defaults for output options
        output_options = {
            'word_timestamps': output_options.get('word_timestamps', True),
            'extra_data': output_options.get('extra_data', True),
        }

        # Define the synchronous function to run in thread
        def run_transcription():
            return list(self.transcribe_core(
                path=path, url=url, blob=blob, language=language,
                diarize=diarize, diarization_args=diarization_args,
                output_options=output_options, verbose=verbose,
                on_progress=on_progress, **kwargs
            ))

        # Run transcription in thread pool to allow other coroutines to continue
        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(None, run_transcription)

        # Yield segments
        for segment in segments:
            yield segment


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
            logger.info(
                f"Session {self.session_id}: Added {len(audio_bytes)} bytes, "
                f"total duration: {self.total_duration:.2f}s"
            )

        # Process buffer to extract any complete segments and accumulate them
        complete_segments = self._transcribe_buffered(flush=False)
        if complete_segments:
            self.accumulated_segments.extend(complete_segments)
            if self.verbose:
                logger.info(
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
            logger.info(f"Session {self.session_id}: Reset")
    
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
                logger.info(f"Session {self.session_id}: Flushed {len(remaining_segments)} final segments")
            
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
                            logger.info(f"Session {self.session_id}: Trimmed {samples_to_remove} samples ({last_complete_end_time:.2f}s)")
                    
                    return complete_segments
                else:
                    # No complete segments yet (0 or 1 segments)
                    return []
            
        except Exception as e:
            logger.error(f"Error during session buffer transcription: {e}")
            return []
        finally:
            # Close the WAV buffer
            wav_buffer.close()


class FasterWhisperModel(TranscriptionModel):
    """Faster Whisper transcription model"""
    
    def __init__(self, model: str, device: str = None, local_files_only: bool = False, **kwargs):
        super().__init__(engine="faster-whisper", model=model)
        
        # Check for required dependencies
        utils.check_dependencies(['faster_whisper', 'numpy'], 'FasterWhisperModel')
        
        self.model_path = model
        self.device = device if device else utils.guess_device()
        self.local_files_only = local_files_only
        self.model_kwargs = kwargs
        
        # Load the model immediately
        self.model_object = self._load_faster_whisper_model()
    
    def _load_faster_whisper_model(self) -> Any:
        """
        Load the actual faster-whisper model.
        """
        # Import faster_whisper
        import faster_whisper
        
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
        
        # Set default compute_type based on device if not provided by user.
        # We have seen cases where transcription accuracy degrades when using int8.
        if 'compute_type' not in self.model_kwargs:
            args['compute_type'] = 'float16' if device == 'cuda' else 'float32'
        
        # Add any additional kwargs passed to the constructor
        args.update(self.model_kwargs)
        
        logger.info(f'Loading faster-whisper model: {self.model_path} on {device} with index: {device_index or 0}')
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
            logger.info(f"Created FasterWhisper transcription session: {session_id}")
        
        return session

    def transcribe_core(
        self,
        *,
        path: Optional[str] = None,
        url: Optional[str] = None,
        blob: Optional[str] = None,
        language: Optional[str] = None,
        diarize: bool = False,
        diarization_args: Optional[Dict[str, Any]] = None,
        output_options: Dict[str, Any],
        verbose: bool = False,
        on_progress: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> Generator[Segment, None, None]:
        """
        Transcribe using faster-whisper engine.
        """
        # Validate diarization support
        if diarize:
            raise NotImplementedError("Diarization (diarize=True) is only supported with StableWhisper models. "
                                    "Please use StableWhisperModel or RunPodModel with core_engine='stable-whisper'.")

        # Handle URL download or blob processing if needed
        audio_path = utils.get_audio_file_path(path=path, url=url, blob=blob, verbose=verbose)

        if verbose:
            logger.info(f"Using faster-whisper engine with model: {self.model}")
            logger.info(f"Processing file: {audio_path}")
            if self.model_object:
                logger.info(f"Using pre-loaded model: {self.model_object}")
            if diarize:
                logger.info("Diarization is enabled")

        try:
            # Transcribe using faster-whisper directly with file path
            # Enable word timestamps if requested
            transcribe_args = {
                'language': language,
                'word_timestamps': output_options['word_timestamps'],
                **kwargs
            }
            segments, info = self.model_object.transcribe(audio_path, **transcribe_args)
            total_seconds = getattr(info, "duration", None)

            # Collect segments for diarization if needed
            all_segments = [] if diarize else None

            for segment in segments:
                # Build extra_data dictionary if requested
                segment_extra_data = _copy_segment_extra_data(segment, language=language) if output_options['extra_data'] else {}

                # Process words if available and requested
                segment_words = []
                if output_options['word_timestamps'] and hasattr(segment, 'words') and segment.words:
                    for word_data in segment.words:
                        word = Word(
                            word=word_data.word,
                            start=word_data.start,
                            end=word_data.end,
                            probability=getattr(word_data, 'probability', None)
                        )
                        segment_words.append(word)

                # Create Segment object
                segment_obj = Segment(
                    text=segment.text,
                    start=segment.start,
                    end=segment.end,
                    words=segment_words,
                    extra_data=segment_extra_data
                )

                emit_progress(
                    on_progress,
                    phase="transcription",
                    step="decode",
                    step_fraction=segment.end / total_seconds if total_seconds else 0.0,
                    description="Transcribing audio",
                    processed_seconds=segment.end,
                    total_seconds=total_seconds,
                )

                if diarize:
                    all_segments.append(segment_obj)
                else:
                    yield segment_obj
            
            # Apply diarization if requested
            if diarize:
                from .diarization import diarize as diarize_func
                
                # Copy user diarization arguments and set defaults
                diar_kwargs = (diarization_args or {}).copy()
                diar_kwargs.setdefault("engine", "ivrit")
                diar_kwargs.setdefault("device", self.device)
                
                all_segments = diarize_func(
                    audio=audio_path,
                    transcription_segments=all_segments,
                    verbose=verbose,
                    on_progress=on_progress,
                    **diar_kwargs
                )
                
                # Yield all segments
                for segment in all_segments:
                    yield segment
                
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise

        finally:
            # Clean up temporary files created for URL downloads or blob processing
            if (url is not None or blob is not None) and os.path.exists(audio_path):
                os.remove(audio_path)


class StableWhisperModel(TranscriptionModel):
    """Stable Whisper transcription model"""
    
    def __init__(self, model: str, device: str = None, local_files_only: bool = False, **kwargs):
        super().__init__(engine="stable-whisper", model=model)
        
        # Check for required dependencies
        utils.check_dependencies(['stable_whisper', 'numpy'], 'StableWhisperModel')
        
        self.model_path = model
        self.device = device if device else utils.guess_device()
        self.local_files_only = local_files_only
        self.model_kwargs = kwargs
        
        # Load the model immediately
        self.model_object = self._load_stable_whisper_model()
    
    def _load_stable_whisper_model(self) -> Any:
        """
        Load the actual stable-whisper model.
        """
        # Import stable_whisper
        import stable_whisper
        
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
        
        # Set default compute_type based on device if not provided by user
        # We have seen cases where transcription accuracy degrades when using int8.
        if 'compute_type' not in self.model_kwargs:
            args['compute_type'] = 'float16' if device == 'cuda' else 'float32'

        # Add any additional kwargs passed to the constructor
        args.update(self.model_kwargs)
        
        logger.info(f'Loading stable-whisper model: {self.model_path} on {device} with index: {device_index or 0}')
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
            logger.info(f"Created StableWhisper transcription session: {session_id}")
        
        return session

    def transcribe_core(
        self,
        *,
        path: Optional[str] = None,
        url: Optional[str] = None,
        blob: Optional[str] = None,
        language: Optional[str] = None,
        diarize: bool = False,
        diarization_args: Optional[Dict[str, Any]] = None,
        output_options: Dict[str, Any],
        verbose: bool = False,
        on_progress: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> Generator[Segment, None, None]:
        """
        Transcribe using stable-whisper engine.
        """
        # Handle URL download or blob processing if needed
        audio_path = utils.get_audio_file_path(path=path, url=url, blob=blob, verbose=verbose)

        if verbose:
            logger.info(f"Using stable-whisper engine with model: {self.model}")
            logger.info(f"Processing file: {audio_path}")
            if self.model_object:
                logger.info(f"Using pre-loaded model: {self.model_object}")
            if diarize:
                logger.info("Diarization is enabled")

        try:
            # Adapt the unified on_progress contract to stable-whisperless's
            # native (seek_seconds, total_duration_seconds) progress_callback.
            # We pass this wrapper unconditionally; emit_progress is a no-op
            # when on_progress is None.
            def _stable_whisper_progress(seek: float, total: float) -> None:
                emit_progress(
                    on_progress,
                    phase="transcription",
                    step="decode",
                    step_fraction=seek / total if total else 0.0,
                    description="Transcribing audio",
                    processed_seconds=float(seek),
                    total_seconds=float(total),
                )

            # Transcribe using stable-whisper with word timestamps
            # Enable word timestamps if requested
            result = self.model_object.transcribe(
                audio_path,
                language=language,
                word_timestamps=output_options['word_timestamps'],
                progress_callback=_stable_whisper_progress,
            )
            segments = result.segments
            
            # Collect segments for diarization if needed
            all_segments = [] if diarize else None
            
            for segment in segments:
                # Build extra_data dictionary if requested
                segment_extra_data = _copy_segment_extra_data(segment, language=language) if output_options['extra_data'] else {}
                
                # Process words if available and requested
                segment_words = []
                if output_options['word_timestamps'] and hasattr(segment, 'words') and segment.words:
                    for word_data in segment.words:
                        word = Word(
                            word=word_data.word,
                            start=word_data.start,
                            end=word_data.end,
                            probability=getattr(word_data, 'probability', None)
                        )
                        segment_words.append(word)
                
                # Create Segment object
                segment_obj = Segment(
                    text=segment.text,
                    start=segment.start,
                    end=segment.end,
                    words=segment_words,
                    extra_data=segment_extra_data
                )
                
                if diarize:
                    all_segments.append(segment_obj)
                else:
                    yield segment_obj
            
            # Apply diarization if requested
            if diarize:
                from .diarization import diarize as diarize_func
                
                # Copy user diarization arguments and set defaults
                diar_kwargs = (diarization_args or {}).copy()
                diar_kwargs.setdefault("engine", "ivrit")
                diar_kwargs.setdefault("device", self.device)
                
                all_segments = diarize_func(
                    audio=audio_path,
                    transcription_segments=all_segments,
                    verbose=verbose,
                    on_progress=on_progress,
                    **diar_kwargs
                )
                
                # Yield all segments
                for segment in all_segments:
                    yield segment
                
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise

        finally:
            # Clean up temporary files created for URL downloads or blob processing
            if (url is not None or blob is not None) and os.path.exists(audio_path):
                os.remove(audio_path)


class WhisperCppModel(TranscriptionModel):
    """whisper.cpp transcription model via pywhispercpp"""
    
    def __init__(self, model: str, n_threads: int = None, **kwargs):
        """
        Initialize a whisper.cpp model.
        
        Args:
            model: Model name (e.g., 'base', 'small', 'medium', 'large') or path to a .bin model file
            n_threads: Number of threads to use for transcription (default: auto-detect)
            **kwargs: Additional arguments passed to pywhispercpp Model constructor
        """
        super().__init__(engine="whisper-cpp", model=model)
        
        # Check for required dependencies
        utils.check_dependencies(['pywhispercpp.model'], 'WhisperCppModel')
        
        self.model_path = model
        self.n_threads = n_threads
        self.model_kwargs = kwargs
        
        # Load the model immediately
        self.model_object = self._load_whisper_cpp_model()
    
    def _load_whisper_cpp_model(self) -> Any:
        """
        Load the actual whisper.cpp model via pywhispercpp.
        """
        from pywhispercpp.model import Model
        
        logger.info(f'Loading whisper.cpp model: {self.model_path}')
        
        args = {}
        if self.n_threads is not None:
            args['n_threads'] = self.n_threads
       
        # Use beam-search by default.
        # User can override this via model_kwargs. 
        args['params_sampling_strategy'] = 1

        # Add any additional kwargs passed to the constructor
        args.update(self.model_kwargs)
        
        return Model(self.model_path, **args)
    
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
            logger.info(f"Created WhisperCpp transcription session: {session_id}")
        
        return session

    def transcribe_core(
        self,
        *,
        path: Optional[str] = None,
        url: Optional[str] = None,
        blob: Optional[str] = None,
        language: Optional[str] = None,
        diarize: bool = False,
        diarization_args: Optional[Dict[str, Any]] = None,
        output_options: Dict[str, Any] = None,
        verbose: bool = False,
        on_progress: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> Generator[Segment, None, None]:
        """
        Transcribe using whisper.cpp engine via pywhispercpp.
        """
        # Default output_options if not provided
        if output_options is None:
            output_options = {'word_timestamps': True, 'extra_data': True}

        # Validate diarization support
        if diarize:
            raise NotImplementedError("Diarization (diarize=True) is not supported with whisper.cpp engine. "
                                    "Please use StableWhisperModel or RunPodModel with core_engine='stable-whisper'.")

        # Handle URL download or blob processing if needed
        audio_path = utils.get_audio_file_path(path=path, url=url, blob=blob, verbose=verbose)

        if verbose:
            logger.info(f"Using whisper.cpp engine with model: {self.model}")
            logger.info(f"Processing file: {audio_path}")

        try:
            # Build transcribe arguments
            transcribe_args = {}
            if language is not None:
                transcribe_args['language'] = language

            # Add any extra kwargs
            transcribe_args.update(kwargs)

            # Adapt the unified on_progress contract to pywhispercpp's
            # new_segment_callback. pywhispercpp passes a Segment namedtuple
            # with t0/t1 in centiseconds. Probe duration via ffmpeg so we
            # can report a meaningful step_fraction.
            if on_progress is not None:
                total_seconds = utils.get_audio_duration(audio_path)
                def _whisper_cpp_segment(segment) -> None:
                    processed = segment.t1 / 100.0
                    emit_progress(
                        on_progress,
                        phase="transcription",
                        step="decode",
                        step_fraction=processed / total_seconds if total_seconds else 0.0,
                        description="Transcribing audio",
                        processed_seconds=processed,
                        total_seconds=total_seconds,
                    )
                transcribe_args["new_segment_callback"] = _whisper_cpp_segment

            # Transcribe using pywhispercpp
            # Returns list of Segment namedtuples with t0, t1 (centiseconds), text
            segments = self.model_object.transcribe(audio_path, **transcribe_args)
            
            for segment in segments:
                # Convert centiseconds to seconds
                # pywhispercpp uses t0/t1 in centiseconds (1/100th of a second)
                start_time = segment.t0 / 100.0
                end_time = segment.t1 / 100.0
                
                # Build extra_data dictionary if requested
                segment_extra_data = {}
                if output_options.get('extra_data', True):
                    if language:
                        segment_extra_data['language'] = language
                
                # Create Segment object
                # Note: pywhispercpp doesn't provide word-level timestamps in basic API
                segment_obj = Segment(
                    text=segment.text,
                    start=start_time,
                    end=end_time,
                    words=[],
                    extra_data=segment_extra_data
                )
                
                yield segment_obj
                
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
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
        self.job_id = None

        logger.debug(f"RunPodJob: POST {self.base_url}/run")
        t0 = time.monotonic()
        response = requests.post(
            f"{self.base_url}/run",
            headers=self.headers,
            json=payload,
        )
        logger.debug(f"RunPodJob: submit returned status={response.status_code} in {time.monotonic()-t0:.2f}s")

        if response.status_code == 401:
            logger.error("RunPod API authentication failed: invalid API key")
            raise Exception("Invalid RunPod API key")

        if not response.ok:
            logger.error(f"RunPod job submission failed: HTTP {response.status_code} {response.reason}")
        response.raise_for_status()

        result = response.json()
        self.job_id = result.get("id")
        logger.debug(f"RunPodJob[{self.job_id}]: submitted")

    def status_body(self):
        """Fetch /status. A failed job's error is reported here and nowhere else."""
        logger.debug(f"RunPodJob[{self.job_id}]: GET /status")
        t0 = time.monotonic()
        response = requests.get(
            f"{self.base_url}/status/{self.job_id}",
            headers=self.headers,
        )
        logger.debug(f"RunPodJob[{self.job_id}]: status HTTP {response.status_code} in {time.monotonic()-t0:.2f}s")
        response.raise_for_status()

        body = response.json()
        logger.debug(f"RunPodJob[{self.job_id}]: status={body.get('status', 'UNKNOWN')}")
        return body

    def _stream_ended(self):
        """Decide what a /stream response that carries no more data means.

        /stream reports COMPLETED for jobs that have actually failed, and for
        jobs that have not started yet, so the job's real outcome has to be read
        back from /status. Returns True when the job is genuinely finished and
        False when streaming should continue; raises if the job failed.
        """
        body = self.status_body()
        job_status = body.get("status", "UNKNOWN")

        if job_status in ('IN_QUEUE', 'IN_PROGRESS'):
            logger.debug(f"RunPodJob[{self.job_id}]: stream ended early, job is {job_status}, continuing")
            time.sleep(1)
            return False

        if job_status != 'COMPLETED':
            logger.error(f"RunPodJob[{self.job_id}]: job {job_status}: {body.get('error')}")
            raise Exception(f"RunPod job {job_status}: {body.get('error')}")

        return True

    def stream(self):
        """Stream job results"""
        iter_n = 0
        while True:
            iter_n += 1
            logger.debug(f"RunPodJob[{self.job_id}]: stream iter={iter_n} GET /stream")
            t0 = time.monotonic()
            response = requests.get(
                f"{self.base_url}/stream/{self.job_id}",
                headers=self.headers,
                stream=True,
            )
            logger.debug(f"RunPodJob[{self.job_id}]: stream iter={iter_n} headers HTTP {response.status_code} in {time.monotonic()-t0:.2f}s")
            response.raise_for_status()

            # Expect a single response
            try:
                t1 = time.monotonic()
                raw = response.content
                logger.debug(f"RunPodJob[{self.job_id}]: stream iter={iter_n} body read bytes={len(raw)} in {time.monotonic()-t1:.2f}s")
                content = raw.decode('utf-8')
                data = json.loads(content)
                job_status = data['status']
                stream_block = data.get('stream', []) or []
                logger.debug(f"RunPodJob[{self.job_id}]: stream iter={iter_n} parsed job_status={job_status} stream_items={len(stream_block)}")

                if job_status not in ['IN_PROGRESS', 'COMPLETED']:
                    logger.debug(f"RunPodJob[{self.job_id}]: stream iter={iter_n} non-progress status={job_status}")
                    if self._stream_ended():
                        return
                    continue

                yielded_segments = 0
                yielded_progress = 0
                for item in data['stream']:
                    if 'output' in item:
                        for entry in item['output']:
                            index = entry.get('index')
                            entry_type = entry.get('type')
                            if entry_type == 'segments':
                                for element in entry['data']:
                                    try:
                                        segment = Segment(**element)
                                    except Exception as e:
                                        logger.error(f"Failed to decode RunPod stream element: {e}")
                                        raise Exception(f"Failed to decode JSON: {e}")
                                    yield segment if index is None else (index, segment)
                                    yielded_segments += 1
                            elif entry_type == 'progress':
                                if index is None:
                                    yield {"progress": entry['data']}
                                else:
                                    yield {"progress": entry['data'], "index": index}
                                yielded_progress += 1
                            elif entry_type == 'error':
                                exc = Exception(entry['data'])
                                yield exc if index is None else (index, exc)

                logger.debug(f"RunPodJob[{self.job_id}]: stream iter={iter_n} yielded segments={yielded_segments} progress={yielded_progress}")

                if data['status'] == 'COMPLETED':
                    logger.debug(f"RunPodJob[{self.job_id}]: stream COMPLETED after iter={iter_n}")
                    if self._stream_ended():
                        return

            except json.JSONDecodeError as e:
                logger.error(f"RunPodJob[{self.job_id}]: failed to parse stream JSON: {e}")
                if self._stream_ended():
                    return

    def cancel(self):
        """Cancel the job"""
        logger.debug(f"RunPodJob[{self.job_id}]: POST /cancel")
        t0 = time.monotonic()
        response = requests.post(
            f"{self.base_url}/cancel/{self.job_id}",
            headers=self.headers,
        )
        logger.debug(f"RunPodJob[{self.job_id}]: cancel HTTP {response.status_code} in {time.monotonic()-t0:.2f}s")
        response.raise_for_status()

        return response.json()

    def get_timings(self):
        """Fetch RunPod billing timings (delayTime/executionTime, in ms) from /status."""
        data = self.status_body()
        return {
            "queue_ms": data.get("delayTime"),
            "execution_ms": data.get("executionTime"),
        }


class AsyncRunPodJob:
    """Async version of RunPodJob for better scalability with high concurrency."""
    
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
                    logger.error("RunPod API authentication failed: invalid API key")
                    raise Exception("Invalid RunPod API key")

                if response.status >= 400:
                    logger.error(f"RunPod async job submission failed: HTTP {response.status}")
                response.raise_for_status()
                result = await response.json()
                self.job_id = result.get("id")
                logger.debug(f"AsyncRunPodJob[{self.job_id}]: submitted")

    async def status_body(self):
        """Fetch /status. A failed job's error is reported here and nowhere else."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/status/{self.job_id}",
                headers=self.headers
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def _stream_ended(self):
        """Decide what a /stream response that carries no more data means.

        /stream reports COMPLETED for jobs that have actually failed, and for
        jobs that have not started yet, so the job's real outcome has to be read
        back from /status. Returns True when the job is genuinely finished and
        False when streaming should continue; raises if the job failed.
        """
        body = await self.status_body()
        job_status = body.get("status", "UNKNOWN")

        if job_status in ('IN_QUEUE', 'IN_PROGRESS'):
            logger.debug(f"AsyncRunPodJob[{self.job_id}]: stream ended early, job is {job_status}, continuing")
            await asyncio.sleep(1)
            return False

        if job_status != 'COMPLETED':
            logger.error(f"AsyncRunPodJob[{self.job_id}]: job {job_status}: {body.get('error')}")
            raise Exception(f"RunPod job {job_status}: {body.get('error')}")

        return True

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
                            if await self._stream_ended():
                                return
                            continue

                        for item in data['stream']:
                            if 'output' in item:
                                for entry in item['output']:
                                    index = entry.get('index')
                                    entry_type = entry.get('type')
                                    if entry_type == 'segments':
                                        for element in entry['data']:
                                            try:
                                                segment = Segment(**element)
                                            except Exception as e:
                                                logger.error(f"Failed to decode RunPod async stream element: {e}")
                                                raise Exception(f"Failed to decode JSON: {e}")
                                            yield segment if index is None else (index, segment)
                                    elif entry_type == 'progress':
                                        if index is None:
                                            yield {"progress": entry['data']}
                                        else:
                                            yield {"progress": entry['data'], "index": index}
                                    elif entry_type == 'error':
                                        exc = Exception(entry['data'])
                                        yield exc if index is None else (index, exc)

                        if data['status'] == 'COMPLETED':
                            if await self._stream_ended():
                                return

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse RunPod async JSON response: {e}")
                        if await self._stream_ended():
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

    async def get_timings(self):
        """Fetch RunPod billing timings (delayTime/executionTime, in ms) from /status."""
        data = await self.status_body()
        return {
            "queue_ms": data.get("delayTime"),
            "execution_ms": data.get("executionTime"),
        }


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
            logger.info(f"Created RunPod transcription session: {session_id}")
            logger.info("Note: RunPod sessions buffer audio locally and transcribe on flush()")
        
        return session

    @staticmethod
    def _payload_byte_len(payload) -> int:
        """Return the JSON UTF-8 byte length of a payload — the size that
        actually goes on the wire, which correctly counts multi-byte
        (e.g. Hebrew) characters that ``len(str(payload))`` would undercount."""
        return len(json.dumps(payload).encode("utf-8"))

    def _encode_sources(self, kind: str, items: List[str]) -> List[str]:
        """Encode each batch element exactly once: path -> read file +
        base64; blob/url -> passthrough. The returned values are reused for
        both bin-packing and payload assembly so a file is never read twice."""
        def encode_element(element: str) -> str:
            if kind == "path":
                try:
                    with open(element, 'rb') as f:
                        audio_data = f.read()
                    return base64.b64encode(audio_data).decode('utf-8')
                except Exception as e:
                    logger.error(f"Failed to read audio file for RunPod: {e}")
                    raise Exception(f"Failed to read audio file: {e}")
            return element

        return [encode_element(element) for element in items]

    def _assemble_payload(
        self,
        *,
        kind: str,
        source_value: Union[str, List[str]],
        language: Optional[str],
        diarize: bool,
        diarization_args: Optional[Dict[str, Any]],
        output_options: Dict[str, Any],
        verbose: bool,
        **kwargs,
    ) -> dict:
        """
        Build the RunPod job payload dict from an already-encoded
        ``source_value`` (a scalar for single-source mode, or a list for batch
        mode). The list-ness of ``source_value`` is preserved into the
        ``transcribe_args`` source key (``url`` / ``blob``); the scalar
        top-level ``type`` field ("blob"/"url") is unaffected by list-ness.
        """
        if kind == "path" or kind == "blob":
            payload_type = "blob"
            source_key = "blob"
        elif kind == "url":
            payload_type = "url"
            source_key = "url"
        else:
            raise ValueError("Must specify either 'path', 'url', or 'blob'")

        return {
            "input": {
                "type": payload_type,
                "model": self.model,
                "engine": self.core_engine,
                "streaming": True,
                "transcribe_args": {
                    "language": language,
                    "diarize": diarize,
                    "diarization_args": diarization_args,
                    "output_options": output_options,
                    "verbose": verbose,
                    source_key: source_value,
                    **kwargs
                }
            }
        }

    @staticmethod
    def _plan_blob_chunks(
        encoded_elements: List[str],
        envelope_overhead: int,
        cap: int,
    ) -> List[List[int]]:
        """
        Plan how to split blob/path batch elements into chunks that each fit
        within ``cap`` JSON UTF-8 bytes, returning a list of chunks where each
        chunk is a list of indices into ``encoded_elements``.

        Greedy first-fit IN ORDER: append items to the current chunk until
        adding the next would exceed ``cap``, then start a new chunk. Input
        order is preserved; the concatenation of all chunks == ``[0..n-1]``.

        Fit model: ``envelope_overhead + sum(per-element cost) <= cap``, where
        ``envelope_overhead`` is the byte length of the assembled payload with
        an EMPTY source list, and per-element cost is the element's JSON-string
        byte length plus the list-separator overhead.

        If a single element alone cannot fit
        (``envelope_overhead + that element's cost > cap``), raises
        ``ValueError`` reporting the offending size and the cap. This is the
        home of the eager raise for an unsplittable single blob.

        ``self``-free (staticmethod) so it is unit-testable without a
        credentialed model.
        """
        # Per-element cost: JSON-string byte length plus a separator (", ").
        def element_cost(element: str) -> int:
            return len(json.dumps(element).encode("utf-8")) + len(b", ")

        chunks: List[List[int]] = []
        current: List[int] = []
        current_size = envelope_overhead

        for i, element in enumerate(encoded_elements):
            cost = element_cost(element)
            if envelope_overhead + cost > cap:
                raise ValueError(
                    f"Single source element length is {cost}, exceeding max "
                    f"payload length of {cap}"
                )
            if current and current_size + cost > cap:
                chunks.append(current)
                current = []
                current_size = envelope_overhead
            current.append(i)
            current_size += cost

        if current:
            chunks.append(current)

        return chunks

    def _plan_batch_chunks(
        self,
        *,
        kind: str,
        encoded: List[str],
        language: Optional[str],
        diarize: bool,
        diarization_args: Optional[Dict[str, Any]],
        output_options: Dict[str, Any],
        verbose: bool,
        **kwargs,
    ) -> List[List[int]]:
        """
        Decide how a batch of already-encoded elements maps to RunPod jobs
        (one chunk == one job), returning chunks of global indices in order.

        URL batches are never split: they stay small and scale well, so the
        whole batch is one chunk. blob/path batches are split via
        ``_plan_blob_chunks`` so each resulting job's payload fits the cap; the
        envelope overhead is measured from an assembled payload with an empty
        source list. A single blob/path element alone over the cap raises
        ``ValueError`` (inside ``_plan_blob_chunks``).
        """
        if kind == "url":
            return [list(range(len(encoded)))]

        empty_payload = self._assemble_payload(
            kind=kind,
            source_value=[],
            language=language,
            diarize=diarize,
            diarization_args=diarization_args,
            output_options=output_options,
            verbose=verbose,
            **kwargs,
        )
        envelope_overhead = self._payload_byte_len(empty_payload)
        return self._plan_blob_chunks(encoded, envelope_overhead, self.RUNPOD_MAX_PAYLOAD_LEN)

    def _build_payload(
        self,
        *,
        kind: str,
        source: Union[str, List[str]],
        language: Optional[str],
        diarize: bool,
        diarization_args: Optional[Dict[str, Any]],
        output_options: Dict[str, Any],
        verbose: bool,
        **kwargs,
    ) -> dict:
        """
        Build the RunPod job payload for either a single source or a list of
        sources.

        ``source`` is a single string (single-source mode) or a list of strings
        (batch mode). The list-ness of ``source`` is preserved into the
        ``transcribe_args`` source key (``url`` / ``blob``), so the worker
        receives one value or a list. For ``kind="path"`` each element is read
        and base64-encoded into a blob; for url/blob it is passed through. The
        scalar top-level ``type`` field ("blob"/"url") is unaffected by
        list-ness.

        The combined payload length is enforced against RUNPOD_MAX_PAYLOAD_LEN
        before returning; this aborts a batch eagerly when the aggregate payload
        exceeds the cap.
        """
        is_list = isinstance(source, list)

        if verbose:
            logger.info(f"Using RunPod engine with model: {self.model}")
            logger.info(f"Data source: {source}")

        if is_list:
            source_value = self._encode_sources(kind, source)
        else:
            source_value = self._encode_sources(kind, [source])[0]

        payload = self._assemble_payload(
            kind=kind,
            source_value=source_value,
            language=language,
            diarize=diarize,
            diarization_args=diarization_args,
            output_options=output_options,
            verbose=verbose,
            **kwargs,
        )

        # Check payload size on the whole (possibly batched) payload.
        payload_len = self._payload_byte_len(payload)
        if payload_len > self.RUNPOD_MAX_PAYLOAD_LEN:
            scope = "batched payload" if is_list else "payload"
            logger.error(f"RunPod {scope} too large: {payload_len} bytes (max {self.RUNPOD_MAX_PAYLOAD_LEN})")
            raise ValueError(f"{scope.capitalize()} length is {payload_len}, exceeding max payload length of {self.RUNPOD_MAX_PAYLOAD_LEN}")

        return payload

    def _run_job_stream(self, payload, on_progress):
        """
        Submit a RunPod job and stream its results, demultiplexing the worker
        stream into ``(index, Segment)`` and ``(index, {"progress": ...})``
        tuples where ``index`` is None in single-source mode.

        Preserves the queue-wait, timeout/retry, cancel, billing and finally
        semantics of the original inline implementation. Per-item worker errors
        (tagged ``error`` entries) surface as ``(index, Exception)`` without
        aborting the rest of the stream; an untagged error surfaces as a bare
        Exception (single-source back-compat).
        """
        # Create and execute RunPod job
        run_request = RunPodJob(self.api_key, self.endpoint_id, payload)

        status = None
        for i in range(self.IN_QUEUE_TIMEOUT):
            status_body = run_request.status_body()
            status = status_body.get("status", "UNKNOWN")
            if status == "IN_QUEUE":
                emit_progress(
                    on_progress,
                    phase="transcription",
                    step="queue",
                    step_fraction=0.0,
                    description="Waiting for GPU worker",
                    queue_seconds=i,
                )
                time.sleep(1)
                continue
            break

        if status == "IN_QUEUE":
            emit_progress(
                on_progress,
                phase="transcription",
                step="queue",
                step_fraction=0.0,
                description="Timed out waiting for GPU worker",
            )
            run_request.cancel()
            run_request = None
            raise TimeoutError("Transcription failed: timed out waiting for GPU worker")
        if status not in ("IN_PROGRESS", "COMPLETED"):
            emit_progress(
                on_progress,
                phase="transcription",
                step="queue",
                step_fraction=0.0,
                description=f"Unexpected job status: {status}",
            )
            run_request.cancel()
            run_request = None
            raise Exception(f"Transcription failed: RunPod job {status}: {status_body.get('error')}")

        # Collect streaming results
        timeouts = 0
        loop_iter = 0
        job_id = run_request.job_id
        while True:
            loop_iter += 1
            logger.debug(f"_run_job_stream[{job_id}]: stream loop iter={loop_iter} timeouts={timeouts}")
            try:
                seg_count = 0
                prog_count = 0
                for stream_item in run_request.stream():
                    index, item = self._demux_stream_item(stream_item)
                    if isinstance(item, Segment):
                        seg_count += 1
                        yield (index, item)
                    elif isinstance(item, dict) and "progress" in item:
                        prog_count += 1
                        yield (index, item)
                    elif isinstance(item, Exception):
                        if index is None:
                            raise Exception(f"RunPod error: {item}")
                        yield (index, item)
                    else:
                        raise Exception(f"RunPod error: {stream_item}")

                # If we get here, streaming is complete
                logger.debug(f"_run_job_stream[{job_id}]: stream() returned cleanly iter={loop_iter} segments={seg_count} progress={prog_count}")

                # Log RunPod billing timings (authoritative — same fields RunPod bills on)
                try:
                    timings = run_request.get_timings()
                    logger.debug(
                        f"RunPod[{job_id}] billing: queue={timings['queue_ms']}ms execution={timings['execution_ms']}ms"
                    )
                except Exception as e:
                    logger.warning(f"RunPod[{job_id}]: failed to fetch billing timings: {e}")

                run_request = None
                break

            except requests.exceptions.ReadTimeout:
                timeouts += 1
                if timeouts > self.MAX_STREAM_TIMEOUTS:
                    logger.error(f"RunPod stream timeouts exceeded maximum ({self.MAX_STREAM_TIMEOUTS})")
                    raise Exception(f"Number of request.stream() timeouts exceeded the maximum ({self.MAX_STREAM_TIMEOUTS})")
                logger.warning(f"RunPod stream timeout {timeouts}/{self.MAX_STREAM_TIMEOUTS}, retrying...")
                continue

            except Exception as e:
                logger.error(f"Exception during RunPod streaming: {e}")
                run_request.cancel()
                run_request = None
                raise Exception(f"Exception during RunPod streaming: {e}")

            finally:
                if run_request:
                    run_request.cancel()

    @staticmethod
    def _demux_stream_item(stream_item):
        """
        Split a stream item into ``(index, item)`` where ``index`` is None for
        untagged (single-source) items and an int for batch-tagged items.
        Untagged items arrive as bare ``Segment`` / ``{"progress": ...}`` /
        ``Exception``; tagged items arrive as ``(index, value)`` tuples (the
        progress dict additionally carries an ``"index"`` key).
        """
        if isinstance(stream_item, tuple):
            return stream_item[0], stream_item[1]
        if isinstance(stream_item, dict) and "index" in stream_item:
            index = stream_item["index"]
            return index, {"progress": stream_item["progress"]}
        return None, stream_item

    def _emit_worker_progress(self, on_progress, worker_progress):
        """Forward a worker-emitted progress dict to on_progress, warning on a
        missing 'phase' key (the worker owns the on_progress contract)."""
        if "phase" not in worker_progress:
            logger.warning(
                "RunPod worker progress event missing 'phase' key: %s",
                worker_progress,
            )
        invoke_progress(on_progress, worker_progress)

    def transcribe_core(
        self,
        *,
        path: Optional[str] = None,
        url: Optional[str] = None,
        blob: Optional[str] = None,
        language: Optional[str] = None,
        diarize: bool = False,
        diarization_args: Optional[Dict[str, Any]] = None,
        output_options: Dict[str, Any],
        verbose: bool = False,
        on_progress: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> Generator[Segment, None, None]:
        """
        Transcribe a single source using RunPod engine.
        """
        # Validate diarization support
        if diarize and self.core_engine != "stable-whisper":
            raise NotImplementedError("Diarization (diarize=True) is only supported with core_engine='stable-whisper'. "
                                    f"Current core_engine is '{self.core_engine}'.")

        # Determine source kind and scalar value
        if path is not None:
            kind, source = "path", path
        elif url is not None:
            kind, source = "url", url
        elif blob is not None:
            kind, source = "blob", blob
        else:
            raise ValueError("Must specify either 'path', 'url', or 'blob'")

        payload = self._build_payload(
            kind=kind,
            source=source,
            language=language,
            diarize=diarize,
            diarization_args=diarization_args,
            output_options=output_options,
            verbose=verbose,
            **kwargs,
        )

        for index, item in self._run_job_stream(payload, on_progress):
            if isinstance(item, Segment):
                yield item
            elif isinstance(item, dict) and "progress" in item:
                self._emit_worker_progress(on_progress, item["progress"])

    def _transcribe_batch(
        self,
        *,
        kind: str,
        items: List[str],
        language: Optional[str] = None,
        stream: bool = False,
        diarize: bool = False,
        diarization_args: Optional[Dict[str, Any]] = None,
        output_options: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        on_progress: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> Union[Generator, List[dict]]:
        """
        RunPod batch override. For URL batches and blob/path batches that fit
        the payload cap this submits ONE job with a list payload. A blob/path
        batch whose combined payload exceeds the cap is split into multiple
        sequential jobs via a greedy in-order chunk planner; each job's
        worker-local indices are remapped back to caller GLOBAL indices so the
        caller-visible contract is identical regardless of job count.
        """
        # Validate diarization support
        if diarize and self.core_engine != "stable-whisper":
            raise NotImplementedError("Diarization (diarize=True) is only supported with core_engine='stable-whisper'. "
                                    f"Current core_engine is '{self.core_engine}'.")

        # Default output options if not provided
        if output_options is None:
            output_options = {}
        output_options = {
            'word_timestamps': output_options.get('word_timestamps', True),
            'extra_data': output_options.get('extra_data', True),
        }

        total = len(items)

        # Encode each item exactly once and plan the chunks (one chunk = one job).
        encoded = self._encode_sources(kind, items)
        chunks = self._plan_batch_chunks(
            kind=kind,
            encoded=encoded,
            language=language,
            diarize=diarize,
            diarization_args=diarization_args,
            output_options=output_options,
            verbose=verbose,
            **kwargs,
        )

        def chunk_payload(chunk: List[int]) -> dict:
            return self._assemble_payload(
                kind=kind,
                source_value=[encoded[g] for g in chunk],
                language=language,
                diarize=diarize,
                diarization_args=diarization_args,
                output_options=output_options,
                verbose=verbose,
                **kwargs,
            )

        if stream:
            def batch_generator():
                for chunk in chunks:
                    payload = chunk_payload(chunk)
                    for local_index, item in self._run_job_stream(payload, on_progress):
                        global_index = chunk[local_index]
                        if isinstance(item, Segment):
                            yield (global_index, item)
                        elif isinstance(item, dict) and "progress" in item:
                            wrapped = self._wrap_progress(on_progress, global_index, total)
                            self._emit_worker_progress(wrapped, item["progress"])
                        elif isinstance(item, Exception):
                            yield (global_index, item)
            return batch_generator()

        # Non-streaming: stream under the hood and collect into List[dict] in
        # input order.
        segments_by_index: Dict[int, List[Segment]] = {i: [] for i in range(total)}
        error_by_index: Dict[int, Exception] = {}
        for chunk in chunks:
            payload = chunk_payload(chunk)
            for local_index, item in self._run_job_stream(payload, on_progress):
                global_index = chunk[local_index]
                if isinstance(item, Segment):
                    segments_by_index[global_index].append(item)
                elif isinstance(item, dict) and "progress" in item:
                    wrapped = self._wrap_progress(on_progress, global_index, total)
                    self._emit_worker_progress(wrapped, item["progress"])
                elif isinstance(item, Exception):
                    error_by_index[global_index] = item

        results: List[dict] = []
        for i in range(total):
            if i in error_by_index:
                results.append({"error": str(error_by_index[i]), "source": kind, "input": items[i]})
            else:
                results.append(self._build_result_dict(segments_by_index[i], language))
        return results

    async def _run_job_stream_async(self, payload, on_progress):
        """
        Async variant of ``_run_job_stream``: submit a RunPod job over native
        aiohttp and stream its results, demultiplexing into ``(index, Segment)``
        and ``(index, {"progress": ...})`` tuples where ``index`` is None in
        single-source mode. Preserves the queue-wait, timeout/retry, cancel,
        billing and finally semantics of the original inline implementation.
        """
        # Create and execute RunPod job using native async
        run_request = AsyncRunPodJob(self.api_key, self.endpoint_id, payload)

        # Submit the job
        await run_request.submit()

        status = None
        for i in range(self.IN_QUEUE_TIMEOUT):
            status_body = await run_request.status_body()
            status = status_body.get("status", "UNKNOWN")
            if status == "IN_QUEUE":
                emit_progress(
                    on_progress,
                    phase="transcription",
                    step="queue",
                    step_fraction=0.0,
                    description="Waiting for GPU worker",
                    queue_seconds=i,
                )
                await asyncio.sleep(1)
                continue
            break

        if status == "IN_QUEUE":
            emit_progress(
                on_progress,
                phase="transcription",
                step="queue",
                step_fraction=0.0,
                description="Timed out waiting for GPU worker",
            )
            await run_request.cancel()
            run_request = None
            raise TimeoutError("Transcription failed: timed out waiting for GPU worker")
        if status not in ("IN_PROGRESS", "COMPLETED"):
            emit_progress(
                on_progress,
                phase="transcription",
                step="queue",
                step_fraction=0.0,
                description=f"Unexpected job status: {status}",
            )
            await run_request.cancel()
            run_request = None
            raise Exception(f"Transcription failed: RunPod job {status}: {status_body.get('error')}")

        # Collect streaming results
        timeouts = 0
        while True:
            try:
                async for stream_item in run_request.stream():
                    index, item = self._demux_stream_item(stream_item)
                    if isinstance(item, Segment):
                        yield (index, item)
                    elif isinstance(item, dict) and "progress" in item:
                        yield (index, item)
                    elif isinstance(item, Exception):
                        if index is None:
                            raise Exception(f"RunPod error: {item}")
                        yield (index, item)
                    else:
                        raise Exception(f"RunPod error: {stream_item}")

                # If we get here, streaming is complete.
                # Log RunPod billing timings (authoritative — same fields RunPod bills on).
                try:
                    timings = await run_request.get_timings()
                    logger.debug(
                        f"RunPod[{run_request.job_id}] billing: queue={timings['queue_ms']}ms execution={timings['execution_ms']}ms"
                    )
                except Exception as e:
                    logger.warning(f"RunPod[{run_request.job_id}]: failed to fetch billing timings: {e}")

                run_request = None
                break
            except aiohttp.ClientError as e:
                timeouts += 1
                if timeouts > self.MAX_STREAM_TIMEOUTS:
                    logger.error(f"RunPod async stream timeouts exceeded maximum ({self.MAX_STREAM_TIMEOUTS})")
                    raise Exception(f"Number of request.stream() timeouts exceeded the maximum ({self.MAX_STREAM_TIMEOUTS})")
                logger.warning(f"RunPod async stream timeout {timeouts}/{self.MAX_STREAM_TIMEOUTS}, retrying...")
                continue

            except Exception as e:
                logger.error(f"Exception during RunPod async streaming: {e}")
                await run_request.cancel()
                run_request = None
                raise Exception(f"Exception during RunPod streaming: {e}")

            finally:
                if run_request:
                    await run_request.cancel()

    async def _transcribe_one_async(
        self,
        *,
        path: Optional[str] = None,
        url: Optional[str] = None,
        blob: Optional[str] = None,
        language: Optional[str] = None,
        diarize: bool = False,
        diarization_args: Optional[Dict[str, Any]] = None,
        output_options: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        on_progress: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> AsyncGenerator[Segment, None]:
        """
        Transcribe a single audio source using RunPod asynchronously with native async I/O.

        This is the RunPod override of the single-source async seam.

        This specialized implementation uses aiohttp for better scalability
        when handling many concurrent requests, avoiding thread pool exhaustion.

        Args:
            path: Path to the audio file to transcribe (mutually exclusive with url and blob)
            url: URL to download and transcribe (mutually exclusive with path and blob)
            blob: Base64 encoded blob data to transcribe (mutually exclusive with path and url)
            language: Language code for transcription (e.g., 'he' for Hebrew, 'en' for English)
            diarize: Whether to enable speaker diarization
            diarization_args: Dictionary of arguments for diarization (engine, device, num_speakers, etc.)
            output_options: Dictionary controlling output verbosity. Supported keys:
                - word_timestamps (bool): Whether to populate word-level timestamps (default: True)
                - extra_data (bool): Whether to populate extra metadata fields (default: True)
            verbose: Whether to enable verbose output
            on_progress: Optional progress callback. See TranscriptionModel.transcribe.
            **kwargs: Additional keyword arguments for the transcription model.
        Returns:
            AsyncGenerator yielding transcription segments

        Raises:
            ValueError: If multiple input sources are provided, or none is provided
            FileNotFoundError: If the specified path doesn't exist
            Exception: For other transcription errors
        """
        # Validate diarization support
        if diarize and self.core_engine != "stable-whisper":
            raise NotImplementedError("Diarization (diarize=True) is only supported with core_engine='stable-whisper'. "
                                    f"Current core_engine is '{self.core_engine}'.")

        # Validate arguments
        provided_args = [arg for arg in [path, url, blob] if arg is not None]
        if len(provided_args) > 1:
            raise ValueError("Cannot specify multiple input sources - path, url, and blob are mutually exclusive")

        if len(provided_args) == 0:
            raise ValueError("Must specify either 'path', 'url', or 'blob'")

        # Default output options if not provided
        if output_options is None:
            output_options = {}

        # Set defaults for output options
        output_options = {
            'word_timestamps': output_options.get('word_timestamps', True),
            'extra_data': output_options.get('extra_data', True),
        }

        # Determine source kind and scalar value
        if path is not None:
            kind, source = "path", path
        elif url is not None:
            kind, source = "url", url
        else:
            kind, source = "blob", blob

        payload = self._build_payload(
            kind=kind,
            source=source,
            language=language,
            diarize=diarize,
            diarization_args=diarization_args,
            output_options=output_options,
            verbose=verbose,
            **kwargs,
        )

        async for index, item in self._run_job_stream_async(payload, on_progress):
            if isinstance(item, Segment):
                yield item
            elif isinstance(item, dict) and "progress" in item:
                self._emit_worker_progress(on_progress, item["progress"])

    async def _transcribe_batch_async(
        self,
        *,
        kind: str,
        items: List[str],
        language: Optional[str] = None,
        diarize: bool = False,
        diarization_args: Optional[Dict[str, Any]] = None,
        output_options: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        on_progress: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> AsyncGenerator[tuple[int, Union[Segment, Exception]], None]:
        """
        RunPod async batch override. URL batches and blob/path batches that fit
        the cap run as ONE job; an overflowing blob/path batch is split into
        multiple sequential jobs via the greedy in-order chunk planner. Each
        job's worker-local indices are remapped to caller GLOBAL indices,
        yielding ``(index, Segment)`` and routing per-index progress to a
        wrapped callback. Per-index worker errors surface as
        ``(index, Exception)`` without aborting the rest of the stream.
        """
        # Validate diarization support
        if diarize and self.core_engine != "stable-whisper":
            raise NotImplementedError("Diarization (diarize=True) is only supported with core_engine='stable-whisper'. "
                                    f"Current core_engine is '{self.core_engine}'.")

        # Default output options if not provided
        if output_options is None:
            output_options = {}
        output_options = {
            'word_timestamps': output_options.get('word_timestamps', True),
            'extra_data': output_options.get('extra_data', True),
        }

        total = len(items)

        encoded = self._encode_sources(kind, items)
        chunks = self._plan_batch_chunks(
            kind=kind,
            encoded=encoded,
            language=language,
            diarize=diarize,
            diarization_args=diarization_args,
            output_options=output_options,
            verbose=verbose,
            **kwargs,
        )

        for chunk in chunks:
            chunk_payload = self._assemble_payload(
                kind=kind,
                source_value=[encoded[g] for g in chunk],
                language=language,
                diarize=diarize,
                diarization_args=diarization_args,
                output_options=output_options,
                verbose=verbose,
                **kwargs,
            )
            async for local_index, item in self._run_job_stream_async(chunk_payload, on_progress):
                global_index = chunk[local_index]
                if isinstance(item, Segment):
                    yield (global_index, item)
                elif isinstance(item, dict) and "progress" in item:
                    wrapped = self._wrap_progress(on_progress, global_index, total)
                    self._emit_worker_progress(wrapped, item["progress"])
                elif isinstance(item, Exception):
                    yield (global_index, item)


def load_model(
    *,
    engine: str,
    model: str,
    **kwargs
) -> TranscriptionModel:
    """
    Load a transcription model for the specified engine and model.
    
    Args:
        engine: Transcription engine to use ('faster-whisper', 'stable-whisper', 'whisper-cpp', 'runpod')
        model: Model name for the selected engine
        **kwargs: Additional arguments for specific engines. Known arguments include:
            - faster-whisper: device, local_files_only, compute_type, and any other arguments accepted by WhisperModel
            - stable-whisper: device, local_files_only, compute_type, and any other arguments accepted by stable_whisper.load_faster_whisper
            - whisper-cpp: n_threads, and any other arguments accepted by pywhispercpp.model.Model
            - runpod: api_key (required), endpoint_id (required), core_engine
                     
            Any additional kwargs not recognized by the model wrapper will be passed directly
            to the underlying model constructor (WhisperModel or stable_whisper.load_faster_whisper).
        
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
    elif engine == "whisper-cpp":
        return WhisperCppModel(model=model, **kwargs)
    elif engine == "runpod":
        return RunPodModel(model=model, **kwargs)
    else:
        raise ValueError(f"Unsupported engine: {engine}. Supported engines: 'faster-whisper', 'stable-whisper', 'whisper-cpp', 'runpod'")
