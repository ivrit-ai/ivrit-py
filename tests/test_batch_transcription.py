import types
from types import SimpleNamespace

from ivrit.audio import FasterWhisperModel


class DummySegment:
    def __init__(self, text: str, start: float, end: float):
        self.text = text
        self.start = start
        self.end = end
        self.words = []


class DummyWhisperModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return iter([DummySegment("hello", 0.0, 1.0)]), SimpleNamespace(duration=1.0)


class DummyBatchedInferencePipeline:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return iter([DummySegment("hello", 0.0, 1.0)]), SimpleNamespace(duration=1.0)


def _install_fake_dependencies(monkeypatch):
    import ivrit.audio as audio_module
    import ivrit.utils as utils_module

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = lambda *args, **kwargs: DummyWhisperModel()
    fake_module.BatchedInferencePipeline = DummyBatchedInferencePipeline

    monkeypatch.setattr(
        utils_module,
        "check_dependencies",
        lambda module_specs, feature_name="": {
            name: object() for name in module_specs
        },
    )
    monkeypatch.setattr(utils_module, "guess_device", lambda: "cpu")
    monkeypatch.setattr(
        utils_module,
        "get_audio_file_path",
        lambda **kwargs: kwargs["path"],
    )
    monkeypatch.setattr(audio_module, "get_device_and_index", lambda device: (device, None))
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", fake_module)


def test_faster_whisper_forwards_extra_transcribe_kwargs(monkeypatch):
    _install_fake_dependencies(monkeypatch)
    model = FasterWhisperModel(model="dummy")

    result = model.transcribe(path="audio.mp3", language="he", beam_size=3, vad_filter=True)

    assert result["text"] == "hello"
    assert model.model_object.calls == [
        (
            "audio.mp3",
            {
                "language": "he",
                "word_timestamps": True,
                "beam_size": 3,
                "vad_filter": True,
            },
        ),
    ]


def test_faster_whisper_uses_batched_pipeline_when_batch_size_requested(monkeypatch):
    _install_fake_dependencies(monkeypatch)
    model = FasterWhisperModel(model="dummy")

    result = model.transcribe(path="audio.mp3", language="he", batch_size=8)

    assert result["text"] == "hello"
    assert isinstance(model._batched_model, DummyBatchedInferencePipeline)
    assert model._batched_model.calls == [
        (
            "audio.mp3",
            {
                "language": "he",
                "word_timestamps": True,
                "batch_size": 8,
            },
        ),
    ]
    assert model.model_object.calls == []
