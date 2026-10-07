"""
Regression test for #26:
  FasterWhisperModel.transcribe_core silently drops **kwargs
  StableWhisperModel.transcribe_core silently drops **kwargs

Mocks utils.get_audio_file_path and the underlying model_object.transcribe()
to capture the actual arguments each receives, then asserts the kwargs reach
the engine call.
"""
import inspect
from unittest.mock import MagicMock, patch

from ivrit.audio import FasterWhisperModel, StableWhisperModel


def _make_engine(engine_cls, captured):
    engine = engine_cls.__new__(engine_cls)
    engine.model_object = MagicMock()

    def fake_transcribe(audio_path, **kw):
        captured.update(kw)
        if engine_cls is FasterWhisperModel:
            return iter([]), MagicMock(duration=0.0, language="he", language_probability=1.0)
        result = MagicMock()
        result.segments = iter([])
        return result

    engine.model_object.transcribe.side_effect = fake_transcribe
    return engine


def _run_with_stubbed_path(engine, **kwargs):
    """transcribe_core normalizes path/url/blob via utils.get_audio_file_path
    which would hit the filesystem. Stub it to return a placeholder path."""
    with patch("ivrit.audio.utils.get_audio_file_path", return_value="/tmp/dummy.mp3"):
        list(engine.transcribe_core(**kwargs))


def test_faster_whisper_forwards_kwargs():
    captured = {}
    engine = _make_engine(FasterWhisperModel, captured)
    _run_with_stubbed_path(
        engine,
        path="dummy.mp3",
        language="he",
        output_options={"word_timestamps": False, "extra_data": False},
        initial_prompt="דנה. אלמוג.",
        hotwords="דנה",
        beam_size=7,
    )
    assert "initial_prompt" in captured, f"initial_prompt not forwarded. captured={captured}"
    assert captured["initial_prompt"] == "דנה. אלמוג."
    assert captured.get("hotwords") == "דנה"
    assert captured.get("beam_size") == 7
    print("FasterWhisperModel: kwargs forwarded OK")


def test_stable_whisper_forwards_kwargs():
    captured = {}
    engine = _make_engine(StableWhisperModel, captured)
    _run_with_stubbed_path(
        engine,
        path="dummy.mp3",
        language="he",
        output_options={"word_timestamps": False, "extra_data": False},
        initial_prompt="דנה",
        beam_size=5,
    )
    assert captured.get("initial_prompt") == "דנה", f"captured={captured}"
    assert captured.get("beam_size") == 5
    print("StableWhisperModel: kwargs forwarded OK")


def test_signature_promise():
    sig = inspect.signature(FasterWhisperModel.transcribe_core)
    assert "kwargs" in sig.parameters, "transcribe_core lost its **kwargs parameter"
    sig2 = inspect.signature(StableWhisperModel.transcribe_core)
    assert "kwargs" in sig2.parameters, "StableWhisperModel.transcribe_core lost its **kwargs parameter"
    print("Signatures still promise **kwargs")


if __name__ == "__main__":
    test_signature_promise()
    test_faster_whisper_forwards_kwargs()
    test_stable_whisper_forwards_kwargs()
    print("All regression tests passed.")
