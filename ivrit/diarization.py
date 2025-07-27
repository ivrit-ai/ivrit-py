"""
Speaker diarization functionality for ivrit.ai
------------------------------------------------------------------------------------------------
This file includes modified code from WhisperX (https://github.com/m-bain/whisperX), originally licensed under the BSD 2-Clause License.
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from pyannote.audio import Pipeline

from .types import Segment
from .utils import SAMPLE_RATE, load_audio

DEFAULT_DIARIZATION_CHECKPOINT = "pyannote/speaker-diarization-3.1"


def match_speaker_to_interval(
    diarization_df: pd.DataFrame,
    start: float,
    end: float,
    fill_nearest: bool = False,
) -> Optional[str]:
    """
    Match the best speaker for a given time interval.
    Note: This function modifies the diarization_df in place.
    
    Args:
        diarization_df: Diarization dataframe with columns ['start', 'end', 'speaker']
        start: Start time of the interval
        end: End time of the interval
        fill_nearest: If True, match speakers even when there's no direct time overlap
        
    Returns:
        The speaker ID with the highest intersection, or None if no match found
    """
    # Calculate intersection and union
    diarization_df["intersection"] = np.minimum(diarization_df["end"], end) - np.maximum(diarization_df["start"], start)
    diarization_df["union"] = np.maximum(diarization_df["end"], end) - np.minimum(diarization_df["start"], start)
    
    # Filter based on fill_nearest flag
    if not fill_nearest:
        tmp_df = diarization_df[diarization_df["intersection"] > 0]
    else:
        tmp_df = diarization_df
    
    speaker = None
    
    if len(tmp_df) > 0:
        # Sum over speakers and get the one with highest intersection
        speaker = tmp_df.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
    
    return speaker


def assign_speakers(
    diarization_df: pd.DataFrame,
    transcription_segments: List[Segment],
    fill_nearest: bool = False,
) -> List[Segment]:
    """
    Assign speakers to words and segments in the transcript.

    Args:
        diarization_df: Diarization dataframe with columns ['start', 'end', 'speaker']
        transcription_segments: List of Segment objects to augment with speaker labels
        fill_nearest: If True, assign speakers even when there's no direct time overlap

    Returns:
        Updated transcription_segments with speaker assignments
    """
    for seg in transcription_segments:
        # assign speaker to segment (if any)
        speaker = match_speaker_to_interval(diarization_df, start=seg.start, end=seg.end, fill_nearest=fill_nearest)
        seg.speaker = speaker

        # assign speaker to words
        if hasattr(seg, "words"):
            for word in seg.words:
                if word["start"]:
                    speaker = match_speaker_to_interval(diarization_df, start=word["start"], end=word["end"], fill_nearest=fill_nearest)
                    word["speaker"] = speaker
                    
    return transcription_segments


def diarize(
    audio: Union[str, npt.NDArray],
    *,
    device: Union[str, torch.device] = "cpu",
    checkpoint_path: Optional[Union[str, Path]] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    use_auth_token: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Perform speaker diarization on the given audio.

    Args:
        audio: Path to the audio file or a NumPy array containing the audio waveform.
        device: Device to run diarization on (e.g., "cpu", "cuda", or torch.device).
        checkpoint_path: Optional path or model name for the diarization model checkpoint.
        num_speakers: Optional exact number of speakers to use for diarization.
        min_speakers: Optional minimum number of speakers to consider.
        max_speakers: Optional maximum number of speakers to consider.
        use_auth_token: Optional authentication token for model download if required.
        verbose: Whether to print verbose output during diarization.

    Returns:
        Diarization dataframe with columns ['segment', 'label', 'speaker', 'start', 'end']
    """
    checkpoint_path = checkpoint_path or DEFAULT_DIARIZATION_CHECKPOINT
    if verbose:
        print(f"Diarizing with {checkpoint_path=}, {device=}, {num_speakers=}, {min_speakers=}, {max_speakers=}")

    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(audio, str):
        audio = load_audio(audio)

    audio_data = {
        "waveform": torch.from_numpy(audio[None, :]),
        "sample_rate": SAMPLE_RATE,
    }
    diarization_pipeline = Pipeline.from_pretrained(checkpoint_path, use_auth_token=use_auth_token).to(device)
    diarization = diarization_pipeline(
        audio_data,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    diarization_df = pd.DataFrame(
        diarization.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    diarization_df["start"] = diarization_df["segment"].apply(lambda x: x.start)
    diarization_df["end"] = diarization_df["segment"].apply(lambda x: x.end)
    if verbose:
        print("Diarization completed successfully")
    return diarization_df


def diarize_segments(
    audio: Union[str, npt.NDArray],
    transcription_segments: List[Segment],
    *,
    device: Union[str, torch.device] = "cpu",
    checkpoint_path: Optional[Union[str, Path]] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    use_auth_token: Optional[str] = None,
    verbose: bool = False,
) -> List[Segment]:
    """
    Perform speaker diarization on the given audio and assign speaker labels to transcription segments.
    This method is essentially a convenience wrapper around the diarize() method, assigning speaker labels to transcription segments in place.

    Args:
        audio: Path to the audio file or a NumPy array containing the audio waveform.
        transcription_segments: List of transcription segments to which speaker labels will be assigned.
        device: Device to run diarization on (e.g., "cpu", "cuda", or torch.device).
        checkpoint_path: Optional path or model name for the diarization model checkpoint.
        num_speakers: Optional exact number of speakers to use for diarization.
        min_speakers: Optional minimum number of speakers to consider.
        max_speakers: Optional maximum number of speakers to consider.
        use_auth_token: Optional authentication token for model download if required.
        verbose: Whether to print verbose output during diarization.

    Returns:
        List of transcription segments with speaker labels assigned.
        The returned list is the same as the input list, but with the speaker labels assigned (i.e., the assignment is done in place).

    """
    diarization_df = diarize(
        audio,
        device=device,
        checkpoint_path=checkpoint_path,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        use_auth_token=use_auth_token,
        verbose=verbose,
    )
    diarized_segments = assign_speakers(diarization_df, transcription_segments)

    return diarized_segments
