"""
Audio transcription utilities.

This module provides higher‑level helpers built on top of the generic
`model.transcribe` method exposed by the various Whisper‑based engines
supported by *ivrit*.  The primary addition for this bounty is the
`transcribe_word_level` function which extracts word‑level timestamps
and confidence scores, enabling downstream streaming or confidence‑
analysis workflows.
"""

from __future__ import annotations

from typing import Any, Dict, List

# The concrete model classes are loaded via `ivrit.load_model` and expose a
# `transcribe` method compatible with the Faster‑Whisper API.  We deliberately
# avoid importing a concrete engine here to keep the module engine‑agnostic.
# The type hint `Any` is used for the model because the exact class varies
# between engines (faster‑whisper, openai‑whisper, etc.).
Model = Any


def _extract_word_info(segment: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Helper that extracts word‑level information from a single transcription
    segment.

    Parameters
    ----------
    segment: dict
        A segment dictionary as returned by Whisper‑compatible engines.  The
        expected shape is::

            {
                "id": int,
                "seek": int,
                "start": float,
                "end": float,
                "text": str,
                "words": [
                    {
                        "word": str,
                        "start": float,
                        "end": float,
                        "confidence": float,
                    },
                    ...
                ],
            }

    Returns
    -------
    List[dict]
        A list of dictionaries, each containing ``word``, ``start``, ``end``,
        and ``confidence`` keys.
    """
    words = segment.get("words", [])
    extracted: List[Dict[str, Any]] = []
    for w in words:
        # Some engines may omit confidence; default to ``None`` in that case.
        extracted.append(
            {
                "word": w.get("word"),
                "start": w.get("start"),
                "end": w.get("end"),
                "confidence": w.get("confidence"),
            }
        )
    return extracted


def transcribe_word_level(
    model: Model,
    path: str,
    *,
    language: str | None = None,
    temperature: float = 0.0,
    **transcribe_kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Transcribe an audio file and return word‑level timestamps together with
    confidence scores.

    This function is a thin wrapper around the underlying model's
    ``transcribe`` method.  It forces the engine to emit word‑level
    timestamps (the flag name differs slightly between engines, but most
    accept ``word_timestamps=True``).  The result is normalised into a flat
    list of dictionaries for easy consumption.

    Parameters
    ----------
    model: Any
        An ivrit model instance returned by :func:`ivrit.load_model`.  The
        instance must implement a ``transcribe`` method compatible with the
        Faster‑Whisper API.
    path: str
        Path to the audio file to be transcribed.
    language: str, optional
        Language hint for the model.  If omitted, the model will attempt to
        auto‑detect.
    temperature: float, default ``0.0``
        Sampling temperature; lower values give more deterministic output.
    **transcribe_kwargs:
        Additional keyword arguments forwarded directly to the engine's
        ``transcribe`` method.

    Returns
    -------
    List[dict]
        A list where each element represents a single word and contains the
        following keys:

        - ``word`` (str): The recognised word text.
        - ``start`` (float): Start time in seconds.
        - ``end`` (float): End time in seconds.
        - ``confidence`` (float | None): Confidence score if the engine
          provides it; otherwise ``None``.

    Raises
    ------
    RuntimeError
        If the underlying engine does not support word‑level timestamps.
    """
    # Most Whisper‑compatible engines expose a ``word_timestamps`` flag.
    # To stay engine‑agnostic we pass it and ignore ``TypeError`` if the
    # engine rejects the argument – in that case we raise a clear error.
    try:
        result = model.transcribe(
            path,
            language=language,
            temperature=temperature,
            word_timestamps=True,
            **transcribe_kwargs,
        )
    except TypeError as exc:
        raise RuntimeError(
            "The selected transcription engine does not support word‑level timestamps."
        ) from exc

    # The result format varies slightly; we normalise it.
    # Faster‑Whisper returns a dict with a ``segments`` key.
    segments = result.get("segments") or result.get("segments", [])
    if not isinstance(segments, list):
        # Defensive fallback – some engines may return a flat list directly.
        segments = [result]

    words: List[Dict[str, Any]] = []
    for seg in segments:
        words.extend(_extract_word_info(seg))

    return words
