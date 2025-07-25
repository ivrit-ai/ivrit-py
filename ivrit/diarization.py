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
        diarization_df["intersection"] = np.minimum(diarization_df["end"], seg.end) - np.maximum(
            diarization_df["start"], seg.start
        )
        diarization_df["union"] = np.maximum(diarization_df["end"], seg.end) - np.minimum(
            diarization_df["start"], seg.start
        )
        # remove no hit, otherwise we look for closest (even negative intersection...)
        if not fill_nearest:
            dia_tmp = diarization_df[diarization_df["intersection"] > 0]
        else:
            dia_tmp = diarization_df
        if len(dia_tmp) > 0:
            # sum over speakers
            speaker = diarization_df.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
            seg.speaker = speaker

        # assign speaker to words
        if hasattr(seg, "words"):
            for word in seg.words:
                if word["start"]:
                    diarization_df["intersection"] = np.minimum(diarization_df["end"], word.end) - np.maximum(
                        diarization_df["start"], word["start"]
                    )
                    diarization_df["union"] = np.maximum(diarization_df["end"], word["end"]) - np.minimum(
                        diarization_df["start"], word["start"]
                    )
                    # remove no hit
                    if not fill_nearest:
                        dia_tmp = diarization_df[diarization_df["intersection"] > 0]
                    else:
                        dia_tmp = diarization_df
                    if len(dia_tmp) > 0:
                        # sum over speakers
                        speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                        word["speaker"] = speaker

    return transcription_segments


def diarize(
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
    diarized_segments = assign_speakers(diarization_df, transcription_segments)

    if verbose:
        print("Diarization completed successfully")
    return diarized_segments
