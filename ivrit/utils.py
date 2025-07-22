import os
import subprocess
import tempfile
import urllib.request
from typing import Optional

import numpy as np
import numpy.typing as npt

SAMPLE_RATE = 16000


def get_audio_file_path(
    path: Optional[str] = None, url: Optional[str] = None, verbose: bool = False
) -> str:
    """
    Get the audio file path.

    Args:
        path: Path to the audio file
        url: URL to the audio file
        verbose: Whether to print verbose output

    Returns:
        The audio file path
    """
    # make sure that only one of path or url is provided
    if path is not None and url is not None:
        raise ValueError(
            "Cannot specify both 'path' and 'url' - they are mutually exclusive"
        )
    if path is None and url is None:
        raise ValueError("Must specify either 'path' or 'url'")

    audio_path = path
    temp_file = None

    if url is not None:
        if verbose:
            print(f"Downloading audio from: {url}")

        temp_file = tempfile.NamedTemporaryFile(suffix=".audio")
        urllib.request.urlretrieve(url, temp_file.name)
        audio_path = temp_file.name

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
