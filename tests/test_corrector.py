from ivrit.corrector import SUPPORTED_ENGINES, correct_transcripts


def test_off_is_passthrough_and_copies():
    src = ["shalom olam", "ze haya nisyon"]
    out = correct_transcripts(src, engine="off")
    assert out == src
    assert out is not src
    src[0] = "mutated"
    assert out[0] == "shalom olam"


def test_rejects_bad_input():
    try:
        correct_transcripts("not a list", engine="off")  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")


def test_unknown_engine():
    try:
        correct_transcripts(["x"], engine="nope")
    except ValueError as exc:
        assert "nope" in str(exc)
        for name in SUPPORTED_ENGINES:
            assert name in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_int8_engine_fails_closed_without_weights(monkeypatch):
    monkeypatch.delenv("IVRIT_CORRECTOR_WEIGHTS", raising=False)
    try:
        correct_transcripts(["x"], engine="t5-small-int8")
    except (ImportError, FileNotFoundError, NotImplementedError):
        return
    raise AssertionError("int8 engine must not silently passthrough")
