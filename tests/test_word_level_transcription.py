import os

import pytest

import ivrit


@pytest.mark.slow  # This test hits the Whisper model; may be slower than unit tests.
def test_word_level_transcription_returns_confidence():
    """
    Verify that ``transcribe_word_level`` returns a non‑empty list of word
    dictionaries, each containing the required keys and a confidence value
    (or ``None`` when not provided).
    """
    # Use the small 10‑second test audio bundled with the repository.
    audio_path = os.path.join(os.path.dirname(__file__), "test_input_10s.mp3")

    # Load a model that supports word‑level timestamps.  The default engine
    # used throughout the test suite is ``faster-whisper``.
    model = ivrit.load_model(
        engine="faster-whisper",
        model="ivrit-ai/whisper-large-v3-turbo-ct2",
        device="cpu",  # Keep resource usage modest for CI.
    )

    words = ivrit.transcribe_word_level(model, audio_path)

    # Basic sanity checks.
    assert isinstance(words, list)
    assert len(words) > 0, "Expected at least one word in the transcription."

    required_keys = {"word", "start", "end", "confidence"}
    for w in words:
        assert isinstance(w, dict)
        assert required_keys.issubset(w.keys())
        assert isinstance(w["word"], str)
        assert isinstance(w["start"], (float, int))
        assert isinstance(w["end"], (float, int))
        # Confidence may be ``None`` if the engine does not provide it.
        assert w["confidence"] is None or isinstance(w["confidence"], (float, int))
