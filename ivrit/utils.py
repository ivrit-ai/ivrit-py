"""
This file includes modified code from WhisperX (https://github.com/m-bain/whisperX), originally licensed under the BSD 2-Clause License.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
import base64
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)


# Type alias for the user-facing progress callback used by both transcription
# and diarization. The dict shape is documented under
# TranscriptionModel.transcribe (and the public diarize() function).
ProgressCallback = Callable[[Dict[str, Any]], None]


def invoke_progress(
    on_progress: Optional[ProgressCallback],
    progress: Dict[str, Any],
) -> None:
    """Invoke a user-supplied progress callback with a pre-built dict.

    Exceptions raised by user code are caught and logged at warning level so
    a faulty callback never aborts a transcription or diarization run.
    """
    if on_progress is None:
        return
    try:
        on_progress(progress)
    except Exception as e:
        logger.warning("on_progress callback raised: %s", e)


def emit_progress(
    on_progress: Optional[ProgressCallback],
    *,
    phase: str,
    step: str,
    step_fraction: float,
    description: str,
    **extras: Any,
) -> None:
    """Build a normalized progress dict and invoke the user callback.

    Convenience wrapper around `invoke_progress` for engine integrations
    that build progress events from raw inputs (seek/total floats, segment
    boundaries, clustering steps, etc.).  Engine-specific data passed via
    ``**extras`` is nested under an ``"extra"`` key in the emitted dict.
    """
    if on_progress is None:
        return
    progress: Dict[str, Any] = {
        "phase": phase,
        "step": step,
        "step_fraction": step_fraction,
        "description": description,
        "extra": dict(extras) if extras else {},
    }
    invoke_progress(on_progress, progress)

SAMPLE_RATE = 16000


def get_ffmpeg_exe() -> str:
    """Return the best available ffmpeg executable path."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def check_dependencies(module_specs: List[str], feature_name: str = "This feature") -> Dict[str, Any]:
    """
    Check if required modules are installed and return them.
    
    Args:
        module_specs: List of module names to import (e.g., ['numpy', 'torch'])
        feature_name: Name of the feature requiring these dependencies
        
    Returns:
        Dictionary mapping module names to imported modules
        
    Raises:
        ImportError: If any required module is missing
    """
    missing = []
    modules = {}
    
    for module_spec in module_specs:
        try:
            # Handle module.submodule imports
            if '.' in module_spec:
                parts = module_spec.split('.')
                module = __import__(module_spec)
                # Navigate to the submodule
                for part in parts[1:]:
                    module = getattr(module, part)
                modules[module_spec] = module
            else:
                modules[module_spec] = __import__(module_spec)
        except ImportError:
            missing.append(module_spec)
    
    if missing:
        raise ImportError(
            f"{feature_name} requires additional dependencies: {', '.join(missing)}. "
            f"Install with: pip install ivrit[all]"
        )
    
    return modules


def get_audio_file_path(
    path: Optional[str] = None, 
    url: Optional[str] = None, 
    blob: Optional[str] = None,
    verbose: bool = False
) -> str:
    """
    Get the audio file path.
    Note: In case of url or blob, the file is downloaded/saved to a temporary file, which is not deleted automatically.
    The caller is responsible for deleting the file after use.

    Args:
        path: Path to the audio file
        url: URL to the audio file
        blob: Base64 encoded blob data
        verbose: Whether to print verbose output

    Returns:
        The audio file path
    """
    # make sure that only one of path, url, or blob is provided
    provided_args = [arg for arg in [path, url, blob] if arg is not None]
    if len(provided_args) > 1:
        raise ValueError(
            "Cannot specify multiple input sources - path, url, and blob are mutually exclusive"
        )
    if len(provided_args) == 0:
        raise ValueError("Must specify either 'path', 'url', or 'blob'")

    audio_path = path

    if url is not None:
        if verbose:
            logger.info(f"Downloading audio from: {url}")

        temp_file = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
        audio_path = temp_file.name
        try:
            urllib.request.urlretrieve(url, audio_path)
        except urllib.error.HTTPError as e:
            logger.error(f"Failed to download audio from URL: HTTP {e.code} {e.reason} - {url}")
            os.remove(audio_path)
            raise RuntimeError(
                f"Failed to download audio from URL (HTTP {e.code} {e.reason}): {url}"
            ) from e
        except urllib.error.URLError as e:
            logger.error(f"Failed to download audio from URL: {e.reason} - {url}")
            os.remove(audio_path)
            raise RuntimeError(
                f"Failed to download audio from URL ({e.reason}): {url}"
            ) from e

    if blob is not None:
        if verbose:
            logger.info("Processing blob data")

        temp_file = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
        audio_path = temp_file.name
        
        try:
            blob_bytes = base64.b64decode(blob)
            with open(audio_path, 'wb') as f:
                f.write(blob_bytes)
        except Exception as e:
            logger.error(f"Failed to decode blob data: {e}")
            raise ValueError(f"Failed to decode blob data: {e}")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    return audio_path


def get_audio_duration(path: str) -> Optional[float]:
    """Return the duration in seconds of an audio file using ffmpeg, or None on failure."""
    try:
        import re
        result = subprocess.run(
            [get_ffmpeg_exe(), "-i", path, "-f", "null", "-"],
            capture_output=True, text=True,
        )
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
        if match:
            h, m, s = match.groups()
            return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        pass
    return None


def load_audio(file: str, sr: int = SAMPLE_RATE) -> npt.NDArray:
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    modules = check_dependencies(['numpy'], 'load_audio')
    np = modules['numpy']
    
    try:
        # Launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Prefer imageio-ffmpeg's bundled binary when available, with the system
        # ffmpeg command as a fallback.
        cmd = [
            get_ffmpeg_exe(),
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to load audio via ffmpeg: {e.stderr.decode()}")
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def guess_device():
    modules = check_dependencies(['torch'], 'guess_device')
    torch = modules['torch']

    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
