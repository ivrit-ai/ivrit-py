"""
This file includes modified code from WhisperX (https://github.com/m-bain/whisperX), originally licensed under the BSD 2-Clause License.
"""
import os
import subprocess
import tempfile
import urllib.request
import base64
from typing import Optional

import numpy as np
import numpy.typing as npt

SAMPLE_RATE = 16000


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
            print(f"Downloading audio from: {url}")

        temp_file = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
        audio_path = temp_file.name
        urllib.request.urlretrieve(url, audio_path)

    if blob is not None:
        if verbose:
            print("Processing blob data")

        temp_file = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
        audio_path = temp_file.name
        
        try:
            blob_bytes = base64.b64decode(blob)
            with open(audio_path, 'wb') as f:
                f.write(blob_bytes)
        except Exception as e:
            raise ValueError(f"Failed to decode blob data: {e}")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    return audio_path


def load_audio(file: str, sr: int = SAMPLE_RATE) -> npt.NDArray[np.float32]:
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
    try:
        # Launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI to be installed.
        cmd = [
            "ffmpeg",
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
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def guess_device():
    import torch

    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

