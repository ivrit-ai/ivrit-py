"""Tests for the faster-whisper language-probe encoder reuse.

The optimization skips one full encoder pass per transcription: faster-whisper
encodes window 0 to detect the language and then encodes the same window again
to start decoding.  These tests use a small fake model so they exercise the
bookkeeping without downloading weights, plus a parity check against the real
model (skipped when it cannot run).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from ivrit import _fw_perf

# The adapter depends on faster-whisper's feature windowing; the package itself
# stays importable without the optional extra.
pytest.importorskip("faster_whisper")


class FakeFeatureExtractor:
    """Minimal stand-in for faster_whisper.feature_extractor.FeatureExtractor."""

    sampling_rate = 16000
    nb_max_frames = 3000
    n_samples = 480000
    feature_size = 80

    def __init__(self, frames):
        self.frames = frames
        self.calls = 0

    def __call__(self, waveform, padding=160, chunk_length=None):
        self.calls += 1
        return self.frames


class FakeInnerModel:
    """The CTranslate2 model: only what the adapter touches."""

    is_multilingual = True

    def __init__(self, language="he"):
        self.language = language
        self.detect_calls = 0

    def detect_language(self, encoder_output):
        self.detect_calls += 1
        return [[(f"<|{self.language}|>", 0.99), ("<|en|>", 0.01)]]


class FakeWhisperModel:
    """Behaves like faster_whisper.WhisperModel for the code paths we exercise."""

    def __init__(self, frames, language="he", multilingual=True):
        self.feature_extractor = FakeFeatureExtractor(frames)
        self.model = FakeInnerModel(language)
        self.model.is_multilingual = multilingual
        self.encode_calls = []
        self.transcribe_languages = []

    def encode(self, features):
        self.encode_calls.append(features)
        return f"encoder-output-{len(self.encode_calls)}"

    def transcribe(self, audio, language=None, **options):
        # Mirrors WhisperModel.transcribe: language detection (when needed) runs
        # one encoder pass, then decoding starts by encoding window 0 again.
        self.transcribe_languages.append(language)
        fe = self.feature_extractor
        frames = fe(audio, chunk_length=options.get("chunk_length"))
        content_frames = int(frames.shape[-1]) - 1
        size = min(int(fe.nb_max_frames), content_frames)
        from faster_whisper.audio import pad_or_trim

        window = np.ascontiguousarray(pad_or_trim(frames[:, :size]), dtype=np.float32)
        if language is None:
            if self.model.is_multilingual:
                detected = _fw_perf._detect_language(self, self.encode(window))
                assert detected is not None
                language = detected[0]
            else:
                language = "en"

        def segments():
            self.encode(window)
            yield {"language": language}

        info = SimpleNamespace(
            language=language, language_probability=1.0, all_language_probs=None
        )
        return segments(), info


def _frames(seconds=10.0):
    n = int(seconds * 100)
    return np.linspace(-1.0, 1.0, 80 * n, dtype=np.float32).reshape(80, n)


def _audio(seconds=10.0):
    return np.zeros(int(16000 * seconds), dtype=np.float32)


class TestEncoderReuse:
    def test_reuses_probe_output_instead_of_encoding_twice(self):
        model = FakeWhisperModel(_frames())

        segments, info = _fw_perf.transcribe(model, _audio(), language=None)
        out = list(segments)

        assert info.language == "he"
        assert out == [{"language": "he"}]
        # One encoder pass, and the feature extractor ran once.
        assert len(model.encode_calls) == 1
        assert model.feature_extractor.calls == 1
        # The language probe is skipped because we now know the language.
        assert model.model.detect_calls == 1
        assert model.transcribe_languages == ["he"]
        # The caller's model must not keep any of the adapter's overrides.
        assert model.encode.__self__ is model

    def test_info_reports_the_detected_language_probability(self):
        model = FakeWhisperModel(_frames())

        segments, info = _fw_perf.transcribe(model, _audio(), language=None)
        list(segments)

        # Pinning the language internally must not leak into the reported info.
        assert info.language == "he"
        assert info.language_probability == pytest.approx(0.99)
        assert [name for name, _ in info.all_language_probs] == ["he", "en"]

    def test_vanilla_path_encodes_twice(self, monkeypatch):
        model = FakeWhisperModel(_frames())
        monkeypatch.setattr(_fw_perf, "_supports_reuse", lambda model, options: False)

        segments, _ = _fw_perf.transcribe(model, _audio(), language=None)
        list(segments)

        # The feature extractor only runs once in faster-whisper; it is the
        # encoder that runs twice (language probe + first decoding step).
        assert len(model.encode_calls) == 2
        assert model.feature_extractor.calls == 1

    def test_language_given_skips_the_probe(self):
        model = FakeWhisperModel(_frames())

        segments, info = _fw_perf.transcribe(model, _audio(), language="he")
        list(segments)

        assert len(model.encode_calls) == 1
        assert model.model.detect_calls == 0
        assert info.language == "he"

    @pytest.mark.parametrize(
        "options",
        [
            {"vad_filter": True},
            {"clip_timestamps": "0,10"},
            {"chunk_length": 30},
            {"multilingual": True},
            {"language_detection_segments": 2},
        ],
    )
    def test_falls_back_for_options_that_change_the_first_window(self, options):
        model = FakeWhisperModel(_frames())

        segments, _ = _fw_perf.transcribe(model, _audio(), language=None, **options)
        list(segments)

        # Vanilla behavior: probe pass + decoding pass.
        assert len(model.encode_calls) == 2

    def test_falls_back_for_english_only_models(self):
        model = FakeWhisperModel(_frames(), multilingual=False)

        segments, _ = _fw_perf.transcribe(model, _audio(), language=None)
        list(segments)

        # No probe runs, so only the decoding pass encodes.
        assert len(model.encode_calls) == 1
        assert model.model.detect_calls == 0

    def test_unexpected_window_falls_back_to_the_real_encoder(self, monkeypatch):
        model = FakeWhisperModel(_frames())
        real_first_window = _fw_perf._first_window

        def shifted_first_window(m, features):
            # The probe encodes a window the decoder will never ask for.
            window = real_first_window(m, features)
            return None if window is None else window + 1.0

        monkeypatch.setattr(_fw_perf, "_first_window", shifted_first_window)
        segments, _ = _fw_perf.transcribe(model, _audio(), language=None)
        list(segments)

        # Probe encode + real encode, i.e. the adapter never feeds the decoder
        # a window it did not actually encode.
        assert len(model.encode_calls) == 2

    def test_probe_output_is_not_shared_between_calls(self):
        model = FakeWhisperModel(_frames())

        for _ in range(2):
            segments, _ = _fw_perf.transcribe(model, _audio(), language=None)
            list(segments)

        assert len(model.encode_calls) == 2
        assert model.feature_extractor.calls == 2


class TestFirstWindow:
    def test_long_audio_uses_the_decoder_window(self):
        model = FakeWhisperModel(_frames(40.0))
        features = model.feature_extractor.frames

        window = _fw_perf._first_window(model, features)

        assert window.shape == (80, 3000)
        assert np.array_equal(window, features[:, :3000])

    def test_short_audio_uses_the_decoder_window_not_the_probe_window(self):
        model = FakeWhisperModel(_frames(10.0))  # 1000 Mel frames
        features = model.feature_extractor.frames

        window = _fw_perf._first_window(model, features)

        assert window.shape == (80, 3000)
        # The decoder drops the last Mel frame; faster-whisper's own probe keeps
        # it.  Reusing the probe's encoder output requires probing this window.
        assert np.array_equal(window[:, :999], features[:, :999])
        assert not window[:, 999:].any()


class TestCachedFeatureExtractor:
    def test_reuses_features_for_the_same_waveform_only(self):
        inner = FakeFeatureExtractor(_frames())
        audio = _audio(1.0)
        features = inner(audio)
        cached = _fw_perf._CachedFeatureExtractor(inner, audio, features)

        assert cached(audio) is features
        assert inner.calls == 1
        # A different waveform must fall through to the real extractor.
        other = np.ones(16000, dtype=np.float32)
        cached(other)
        assert inner.calls == 2

    def test_delegates_attributes(self):
        inner = FakeFeatureExtractor(_frames())
        cached = _fw_perf._CachedFeatureExtractor(inner, None, None)
        assert cached.sampling_rate == inner.sampling_rate
        assert cached.nb_max_frames == inner.nb_max_frames


@pytest.fixture(scope="module")
def real_model():
    """A real faster-whisper model, or skip when this machine cannot run one."""
    faster_whisper = pytest.importorskip("faster_whisper")
    try:
        model = faster_whisper.WhisperModel(
            "ivrit-ai/whisper-large-v3-turbo-ct2", device="cuda", compute_type="float16"
        )
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"faster-whisper model unavailable: {exc}")
    yield model


class TestAgainstRealModel:
    """End-to-end parity check against the real faster-whisper engine."""

    def test_optimized_output_matches_vanilla(self, real_model):
        from pathlib import Path

        path = str(Path(__file__).parent / "test_input_10s.mp3")

        def collect(segments):
            return [
                (round(s.start, 3), round(s.end, 3), s.text,
                 tuple((w.word, round(w.start, 3), round(w.end, 3)) for w in (s.words or [])))
                for s in segments
            ]

        vanilla, vanilla_info = real_model.transcribe(
            path, language=None, word_timestamps=True
        )
        optimized, optimized_info = _fw_perf.transcribe(
            real_model, path, language=None, word_timestamps=True
        )

        vanilla_out = collect(vanilla)
        optimized_out = collect(optimized)

        assert optimized_out
        assert optimized_out == vanilla_out
        assert optimized_info.language == vanilla_info.language
