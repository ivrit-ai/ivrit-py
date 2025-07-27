from pathlib import Path

import pytest
from dotenv import load_dotenv

from ivrit.audio import load_model, TranscriptionModel

TEST_INPUT_PATH = str(Path(__file__).parent / "test_input_10s.mp3")
MODEL_NAME = "large-v3-turbo"
LANGUAGE = "he"

@pytest.fixture(scope="session", autouse=True)
def setup_session():
    load_dotenv()

@pytest.fixture(scope="module", params=["faster-whisper", "stable-whisper"])
def model(request: pytest.FixtureRequest) -> TranscriptionModel:
    engine = request.param
    model_ = load_model(engine=engine, model=MODEL_NAME, device="cpu")
    return model_


def test_models_with_diarization(model: TranscriptionModel):
    result = model.transcribe(path=TEST_INPUT_PATH, language=LANGUAGE, diarize=True)
    for seg in result["segments"]:
        assert seg.speaker is not None
    assert result["engine"] == model.engine
    assert result["language"] == LANGUAGE
    assert result["model"] == model.model
