import time

from ivrit.audio import RunPodModel
from ivrit.types import Segment


def test_runpod_model_enables_persistent_session_by_default():
    model = RunPodModel(
        model="ivrit-ai/whisper-large-v3-turbo-ct2",
        api_key="test-key",
        endpoint_id="endpoint-123",
    )

    assert model.use_persistent_session is True
    assert model._session_manager is not None
    model.close()


def test_runpod_session_manager_pings_latest_job(monkeypatch):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append((url, headers, timeout))

        class Response:
            def raise_for_status(self):
                return None

        return Response()

    monkeypatch.setattr("ivrit.audio.requests.get", fake_get)

    model = RunPodModel(
        model="ivrit-ai/whisper-large-v3-turbo-ct2",
        api_key="test-key",
        endpoint_id="endpoint-123",
        keep_alive_interval=0.01,
    )

    model._session_manager.note_job("job-42")
    model._session_manager.start()
    time.sleep(0.05)
    model.close()

    assert any("/status/job-42" in url for url, _, _ in calls)


def test_runpod_transcribe_records_latest_job(monkeypatch):
    class FakeRunPodJob:
        def __init__(self, api_key, endpoint_id, payload):
            self.job_id = "job-99"

        def status(self):
            return "COMPLETED"

        def stream(self):
            yield Segment(text="shalom", start=0.0, end=1.0)

        def cancel(self):
            return {}

    monkeypatch.setattr("ivrit.audio.RunPodJob", FakeRunPodJob)

    model = RunPodModel(
        model="ivrit-ai/whisper-large-v3-turbo-ct2",
        api_key="test-key",
        endpoint_id="endpoint-123",
    )

    segments = list(
        model.transcribe_core(
            blob="ZmFrZQ==",
            language="he",
            output_options={"word_timestamps": True, "extra_data": True},
            verbose=False,
        )
    )
    model.close()

    assert [segment.text for segment in segments] == ["shalom"]
    assert model._session_manager.last_job_id == "job-99"
