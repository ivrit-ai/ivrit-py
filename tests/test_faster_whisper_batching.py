"""
Unit tests for the faster-whisper batched transcription path.

These tests exercise the wrapper logic without downloading a real model:
a fake ``faster_whisper`` module is installed for the duration of each test so
we can assert which backend entry point is selected and which keyword
arguments reach it.
"""

import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

from ivrit.audio import FasterWhisperModel


class _DummyWord:
    def __init__(self, word, start, end, probability=0.9):
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


class _DummySegment:
    def __init__(self, text="hello", start=0.0, end=1.0):
        self.text = text
        self.start = start
        self.end = end
        self.words = [_DummyWord("hello", 0.0, 0.5)]


class _DummyWhisperModel:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return iter([_DummySegment()]), SimpleNamespace(duration=1.0)


class _DummyBatchedInferencePipeline:
    instances = []

    def __init__(self, model):
        self.model = model
        self.calls = []
        _DummyBatchedInferencePipeline.instances.append(self)

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return iter([_DummySegment()]), SimpleNamespace(duration=1.0)


@pytest.fixture
def fake_faster_whisper(monkeypatch):
    """Install fake faster-whisper dependencies and return the fake module."""
    import ivrit.audio as audio_module
    import ivrit.utils as utils_module

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = _DummyWhisperModel
    fake_module.BatchedInferencePipeline = _DummyBatchedInferencePipeline

    monkeypatch.setattr(
        utils_module,
        "check_dependencies",
        lambda module_specs, feature_name="": {name: object() for name in module_specs},
    )
    monkeypatch.setattr(utils_module, "guess_device", lambda: "cpu")
    monkeypatch.setattr(
        utils_module,
        "get_audio_file_path",
        lambda **kwargs: kwargs.get("path") or kwargs.get("url") or kwargs.get("blob"),
    )
    monkeypatch.setattr(audio_module, "get_device_and_index", lambda device: (device, None))
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    _DummyBatchedInferencePipeline.instances = []
    return fake_module


def test_plain_path_forwards_extra_kwargs(fake_faster_whisper):
    model = FasterWhisperModel(model="dummy")

    result = model.transcribe(
        path="audio.mp3",
        language="he",
        beam_size=3,
        vad_filter=True,
        initial_prompt="דנה",
    )

    assert result["text"] == "hello"
    assert model.model_object.calls == [
        (
            "audio.mp3",
            {
                "language": "he",
                "word_timestamps": True,
                "beam_size": 3,
                "vad_filter": True,
                "initial_prompt": "דנה",
            },
        )
    ]


def test_batch_size_selects_batched_pipeline(fake_faster_whisper):
    model = FasterWhisperModel(model="dummy")

    result = model.transcribe(path="audio.mp3", language="he", batch_size=8)

    assert result["text"] == "hello"
    assert model.model_object.calls == []
    assert len(_DummyBatchedInferencePipeline.instances) == 1

    pipeline = _DummyBatchedInferencePipeline.instances[0]
    assert pipeline.model is model.model_object
    assert pipeline.calls == [
        (
            "audio.mp3",
            {
                "language": "he",
                "word_timestamps": True,
                "batch_size": 8,
                # Overrides the batched pipeline's lossy default so ivrit keeps
                # exposing segment timestamps.
                "without_timestamps": False,
            },
        )
    ]


def test_batched_pipeline_respects_explicit_timestamp_flag(fake_faster_whisper):
    model = FasterWhisperModel(model="dummy")

    model.transcribe(path="audio.mp3", batch_size=4, without_timestamps=True)

    pipeline = _DummyBatchedInferencePipeline.instances[0]
    assert pipeline.calls[0][1]["without_timestamps"] is True


def test_batch_size_one_uses_plain_path_without_leaking_kwarg(fake_faster_whisper):
    model = FasterWhisperModel(model="dummy")

    model.transcribe(path="audio.mp3", batch_size=1)

    assert _DummyBatchedInferencePipeline.instances == []
    assert model.model_object.calls == [
        ("audio.mp3", {"language": None, "word_timestamps": True})
    ]


def test_invalid_batch_size_raises(fake_faster_whisper):
    model = FasterWhisperModel(model="dummy")

    with pytest.raises(ValueError, match="batch_size"):
        model.transcribe(path="audio.mp3", batch_size="many")


def test_batch_workers_runs_files_concurrently_in_input_order(fake_faster_whisper):
    model = FasterWhisperModel(model="dummy")

    thread_ids = set()
    lock = threading.Lock()

    def fake_transcribe_one(**kwargs):
        with lock:
            thread_ids.add(threading.get_ident())
        time.sleep(0.05)
        source = kwargs.get("path")
        return {"text": source, "segments": [], "language": "he", "engine": "fake", "model": "fake"}

    model._transcribe_one = fake_transcribe_one

    items = [f"file{i}.mp3" for i in range(6)]
    results = model._transcribe_batch(kind="path", items=items, batch_workers=4)

    assert [r["text"] for r in results] == items
    assert len(thread_ids) > 1, "Expected several worker threads to run the batch"


def test_batch_workers_isolates_failures(fake_faster_whisper):
    model = FasterWhisperModel(model="dummy")

    def fake_transcribe_one(**kwargs):
        source = kwargs.get("path")
        if source == "bad.mp3":
            raise RuntimeError("boom")
        return {"text": source, "segments": [], "language": "he", "engine": "fake", "model": "fake"}

    model._transcribe_one = fake_transcribe_one

    results = model._transcribe_batch(
        kind="path", items=["a.mp3", "bad.mp3", "c.mp3"], batch_workers=3
    )

    assert results[0]["text"] == "a.mp3"
    assert results[1]["error"] == "boom"
    assert results[1]["source"] == "path"
    assert results[1]["input"] == "bad.mp3"
    assert results[2]["text"] == "c.mp3"


def test_batch_workers_stays_sequential_for_streaming(fake_faster_whisper):
    model = FasterWhisperModel(model="dummy")

    seen_kwargs = []

    def fake_transcribe_one(**kwargs):
        seen_kwargs.append(kwargs)
        # League of its own: the top-level transcribe() passes stream through.
        return iter([])

    model._transcribe_one = fake_transcribe_one

    generator = model._transcribe_batch(
        kind="path", items=["a.mp3", "b.mp3"], stream=True, batch_workers=8
    )
    assert list(generator) == []
    # Sequential path was used - no worker kwarg leaked to the engine.
    assert [kw.get("path") for kw in seen_kwargs] == ["a.mp3", "b.mp3"]


def test_batch_workers_does_not_leak_to_backend(fake_faster_whisper):
    model = FasterWhisperModel(model="dummy")

    model.transcribe(path=["a.mp3", "b.mp3"], batch_workers=2)

    for audio, kwargs in model.model_object.calls:
        assert "batch_workers" not in kwargs
