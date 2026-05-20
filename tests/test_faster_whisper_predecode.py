from ivrit.audio import FasterWhisperModel


class FakeInfo:
    duration = 1.0


class FakeSegment:
    text = "hello"
    start = 0.0
    end = 1.0
    words = []


class FakeWhisperModel:
    def __init__(self):
        self.audio_inputs = []

    def transcribe(self, audio, **kwargs):
        self.audio_inputs.append(audio)
        return iter([FakeSegment()]), FakeInfo()


def build_model(fake_backend):
    model = FasterWhisperModel.__new__(FasterWhisperModel)
    model.engine = "faster-whisper"
    model.model = "tiny"
    model.model_object = fake_backend
    model.predecode_audio = True
    model.device = "cpu"
    return model


def test_faster_whisper_predecodes_audio_before_transcribing(monkeypatch, tmp_path):
    audio_path = tmp_path / "input.mp3"
    audio_path.write_bytes(b"fake-audio")
    decoded_audio = object()
    fake_backend = FakeWhisperModel()
    model = build_model(fake_backend)

    monkeypatch.setattr("ivrit.audio.utils.load_audio", lambda path: decoded_audio)

    segments = list(
        model.transcribe_core(
            path=str(audio_path),
            output_options={"word_timestamps": False, "extra_data": False},
        )
    )

    assert len(segments) == 1
    assert fake_backend.audio_inputs == [decoded_audio]


def test_faster_whisper_falls_back_to_path_when_predecode_fails(monkeypatch, tmp_path):
    audio_path = tmp_path / "input.mp3"
    audio_path.write_bytes(b"fake-audio")
    fake_backend = FakeWhisperModel()
    model = build_model(fake_backend)

    def fail_load_audio(path):
        raise RuntimeError("ffmpeg unavailable")

    monkeypatch.setattr("ivrit.audio.utils.load_audio", fail_load_audio)

    segments = list(
        model.transcribe_core(
            path=str(audio_path),
            output_options={"word_timestamps": False, "extra_data": False},
        )
    )

    assert len(segments) == 1
    assert fake_backend.audio_inputs == [str(audio_path)]


def test_faster_whisper_can_disable_predecode(monkeypatch, tmp_path):
    audio_path = tmp_path / "input.mp3"
    audio_path.write_bytes(b"fake-audio")
    fake_backend = FakeWhisperModel()
    model = build_model(fake_backend)
    model.predecode_audio = False

    def fail_if_called(path):
        raise AssertionError("load_audio should not be called")

    monkeypatch.setattr("ivrit.audio.utils.load_audio", fail_if_called)

    segments = list(
        model.transcribe_core(
            path=str(audio_path),
            output_options={"word_timestamps": False, "extra_data": False},
        )
    )

    assert len(segments) == 1
    assert fake_backend.audio_inputs == [str(audio_path)]
