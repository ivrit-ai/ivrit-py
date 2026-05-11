import pytest

from ivrit.audio import FasterWhisperModel, TranscriptionModel, WhisperSession
from ivrit.types import Segment


class RecordingModel(TranscriptionModel):
    def __init__(self):
        super().__init__(engine="recording", model="test")
        self.pcm_calls = []
        self.core_calls = []

    def transcribe_core(self, **kwargs):
        self.core_calls.append(kwargs)
        yield Segment(text="fallback", start=0.0, end=1.0)

    def transcribe_pcm(self, **kwargs):
        self.pcm_calls.append(kwargs)
        yield Segment(text="complete", start=0.0, end=0.5)
        yield Segment(text="pending", start=0.5, end=1.0)


class FallbackModel(TranscriptionModel):
    def __init__(self):
        super().__init__(engine="fallback", model="test")
        self.core_calls = []

    def transcribe_core(self, **kwargs):
        self.core_calls.append(kwargs)
        yield Segment(text="fallback", start=0.0, end=1.0)


class DummyWhisperSegment:
    text = "hello"
    start = 0.0
    end = 0.5
    words = []


class DummyWhisperInfo:
    duration = 0.5


class DummyWhisperModel:
    def __init__(self):
        self.audio = None
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.audio = audio
        self.kwargs = kwargs
        return iter([DummyWhisperSegment()]), DummyWhisperInfo()


def test_session_uses_pcm_transcription_hook():
    model = RecordingModel()
    session = WhisperSession(
        session_id="test-session",
        model=model,
        language="he",
        sample_rate=16000,
    )

    # 0.75s of silence: enough to pass the session's minimum buffer duration.
    session.append(b"\x00\x00" * 12000)

    assert len(model.pcm_calls) == 1
    assert model.core_calls == []
    assert model.pcm_calls[0]["pcm_bytes"] == b"\x00\x00" * 12000
    assert model.pcm_calls[0]["sample_rate"] == 16000
    assert model.pcm_calls[0]["output_options"] == {
        "word_timestamps": True,
        "extra_data": True,
    }
    assert [segment.text for segment in session.get_all_segments()] == ["complete"]


def test_default_pcm_transcription_wraps_pcm_as_blob():
    model = FallbackModel()

    segments = list(model.transcribe_pcm(
        pcm_bytes=b"\x00\x00" * 16000,
        sample_rate=16000,
        language="he",
    ))

    assert [segment.text for segment in segments] == ["fallback"]
    assert len(model.core_calls) == 1
    assert "path" not in model.core_calls[0]
    assert model.core_calls[0]["blob"]
    assert model.core_calls[0]["language"] == "he"
    assert model.core_calls[0]["output_options"] == {
        "word_timestamps": True,
        "extra_data": True,
    }


def test_faster_whisper_pcm_path_passes_decoded_waveform():
    np = pytest.importorskip("numpy")
    whisper = DummyWhisperModel()
    model = FasterWhisperModel.__new__(FasterWhisperModel)
    model.model_object = whisper
    model.model = "dummy"

    segments = list(model.transcribe_pcm(
        pcm_bytes=b"\x00\x00\x00\x40",
        sample_rate=16000,
        language="he",
        output_options={"word_timestamps": False},
    ))

    assert [segment.text for segment in segments] == ["hello"]
    assert isinstance(whisper.audio, np.ndarray)
    assert whisper.audio.dtype == np.float32
    assert whisper.audio.tolist() == [0.0, 0.5]
    assert whisper.kwargs == {
        "language": "he",
        "word_timestamps": False,
    }
