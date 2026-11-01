"""Performance helpers for the faster-whisper engine.

faster-whisper's ``WhisperModel.transcribe()`` runs a full encoder pass to
detect the spoken language when ``language`` is not pinned, and then encodes the
same window again to start decoding.  The probe output is discarded, so that
first encoder pass is pure duplicate work.

This module wraps ``WhisperModel.transcribe()`` and reuses the probe's encoder
output (and the log-Mel features it was computed from) for the first decoding
step.  It is a drop-in replacement for the call in
``FasterWhisperModel.transcribe_core``: it accepts the same audio input and
options, and returns the same ``(segments, info)`` tuple.  Transcripts are
identical because the decoder consumes the output of the same encoder window.

The adapter works on a shallow copy of the model, shadowing ``encode`` and
``feature_extractor`` so the caller's instance is never mutated.  It falls back
to the plain call when an option changes the first window (VAD,
``clip_timestamps``, ``chunk_length``, ``multilingual``, multi-segment language
detection), when the model is not multilingual, or when faster-whisper's
internals are not shaped the way the adapter expects.

See ``PERFORMANCE.md`` for the measurements and the issue #15 investigation.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Tuple

# faster-whisper's ``WhisperModel.transcribe`` uses this sentinel for "no
# explicit clip boundaries".  Anything else changes the first encoder window.
_DEFAULT_CLIP_TIMESTAMPS = "0"


class _CachedFeatureExtractor:
    """Delegating feature extractor that returns already-computed features.

    ``WhisperModel.transcribe`` calls ``self.feature_extractor(audio, ...)``.
    When that call is for the exact waveform we already turned into features,
    reuse the result instead of running the STFT/mel pipeline a second time.
    Everything else (``sampling_rate``, ``nb_max_frames``, ...) is delegated to
    the real extractor, and any unexpected call falls through to it as well.
    """

    def __init__(self, inner: Any, waveform: Any, features: Any) -> None:
        self._inner = inner
        self._waveform = waveform
        self._features = features
        self._used = False

    def __call__(
        self, waveform: Any, padding: int = 160, chunk_length: Optional[int] = None
    ) -> Any:
        # The defaults mirror FeatureExtractor.__call__ so a call that omits them
        # is still recognized as the window we pre-computed.
        if not self._used and chunk_length is None and waveform is self._waveform:
            self._used = True
            return self._features
        return self._inner(waveform, padding=padding, chunk_length=chunk_length)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _supports_reuse(model: Any, options: Dict[str, Any]) -> bool:
    """Return True when the language probe and the decoder see the same window.

    Every option checked here changes which audio the first decoding step sees,
    so the probe's encoder output would not be valid for it.
    """
    if options.get("vad_filter"):
        # VAD trims silence before the encoder, so the decoder window is not the
        # audio we would probe ahead of time.
        return False
    if options.get("clip_timestamps", _DEFAULT_CLIP_TIMESTAMPS) != _DEFAULT_CLIP_TIMESTAMPS:
        return False
    if options.get("chunk_length") is not None:
        return False
    if options.get("multilingual"):
        # The decoder re-runs language detection per window in this mode.
        return False
    if options.get("language_detection_segments", 1) != 1:
        # Detection would encode several windows and vote; we only reuse one.
        return False

    # The internals the adapter relies on.  Keeping this explicit means a
    # future faster-whisper release falls back instead of misbehaving.
    if not hasattr(model, "encode") or not hasattr(model, "feature_extractor"):
        return False
    if not hasattr(model, "model"):
        return False
    inner = model.model
    if not hasattr(inner, "detect_language") or not hasattr(inner, "is_multilingual"):
        return False
    if not inner.is_multilingual:
        # English-only models never run a language probe.
        return False
    if not hasattr(model.feature_extractor, "nb_max_frames"):
        return False
    return True


def _first_window(model: Any, features: Any) -> Optional[Any]:
    """Build the exact feature window that decoding starts with.

    ``WhisperModel.generate_segments`` takes ``features[:, seek:seek+size]``
    with ``seek == 0`` and pads/trims it to one encoder window.  Reproducing
    that here lets the probe's encoder output be the input the decoder would
    otherwise compute.
    """
    import numpy as np
    from faster_whisper.audio import pad_or_trim

    # generate_segments() drops the trailing frame of the feature matrix.
    content_frames = int(features.shape[-1]) - 1
    segment_size = min(int(model.feature_extractor.nb_max_frames), content_frames)
    if segment_size <= 0:
        return None
    window = pad_or_trim(features[:, :segment_size])
    return np.ascontiguousarray(window, dtype=np.float32)


def _detect_language(
    model: Any, encoder_output: Any
) -> Optional[Tuple[str, float, List[Tuple[str, float]]]]:
    """Read the language out of an encoder output, faster-whisper style.

    Returns the same triple ``detect_language()`` builds: language code,
    probability and the full ranked list.  Language tokens are spelled
    ``"<|he|>"``; the delimiters are stripped exactly like faster-whisper does.
    """
    results = model.model.detect_language(encoder_output)[0]
    if not results:
        return None
    all_language_probs = [(token[2:-2], probability) for token, probability in results]
    language, probability = all_language_probs[0]
    return language, probability, all_language_probs


def _same_window(candidate: Any, expected: Any) -> bool:
    """True when ``candidate`` is the probe window we already encoded."""
    import numpy as np

    return (
        isinstance(candidate, np.ndarray)
        and candidate.shape == expected.shape
        and candidate.dtype == expected.dtype
        and np.array_equal(candidate, expected)
    )


def transcribe(
    model: Any,
    audio: Any,
    *,
    language: Optional[str],
    **options: Any,
) -> Tuple[Iterable[Any], Any]:
    """``model.transcribe(audio, ...)`` without the duplicate encoder pass.

    ``audio`` is anything ``WhisperModel.transcribe`` accepts.  It is decoded
    once here (as faster-whisper would do internally) when the probe reuse
    applies.

    Returns the same ``(segments, info)`` tuple as ``model.transcribe``.
    """
    if language is not None or not _supports_reuse(model, options):
        return model.transcribe(audio, language=language, **options)

    import numpy as np
    from faster_whisper.audio import decode_audio

    if not isinstance(audio, np.ndarray):
        audio = decode_audio(audio, sampling_rate=model.feature_extractor.sampling_rate)

    features = model.feature_extractor(audio)
    window = _first_window(model, features)
    if window is None:
        return model.transcribe(audio, language=language, **options)

    # One encoder pass, used both for language detection and for the first
    # decoding step.  This is the pass faster-whisper would otherwise run twice.
    probe_output = model.encode(window)
    detected = _detect_language(model, probe_output)
    if detected is None:
        return model.transcribe(audio, language=language, **options)
    detected_language, language_probability, all_language_probs = detected

    # Shadow the two collaborators on a shallow copy so the caller's model is
    # never mutated (and concurrent transcriptions cannot see our overrides).
    runner = copy.copy(model)
    runner.feature_extractor = _CachedFeatureExtractor(model.feature_extractor, audio, features)
    original_encode = model.encode
    pending_window = window
    pending_output = probe_output

    def encode(feats: Any) -> Any:
        nonlocal pending_window, pending_output
        if pending_window is not None and _same_window(feats, pending_window):
            output = pending_output
            pending_window = pending_output = None
            return output
        return original_encode(feats)

    runner.encode = encode

    # ``transcribe`` is lazy: the decoder pulls features and encoder outputs
    # from ``runner`` only while the returned generator is being consumed.
    segments, info = runner.transcribe(audio, language=detected_language, **options)

    # Pinning ``language`` makes faster-whisper report probability 1 and no
    # ranked list.  Put back what the probe actually measured so ``info`` looks
    # like the auto-detecting call the caller asked for.
    info.language_probability = language_probability
    info.all_language_probs = all_language_probs
    return segments, info
