import pytest
from unittest.mock import MagicMock, patch
from ivrit.client import RunPodClient, LLMProvider, RunPodLLMProvider, create_llm_provider


class TestPersistentSession:
    def test_non_persistent_closes_session(self):
        client = RunPodClient(api_key="test-key", endpoint="https://test.runpod.net")
        assert not client.persistent
        client.close()
        assert client.session.is_closed

    def test_persistent_keeps_session_open(self):
        client = RunPodClient(
            api_key="test-key",
            endpoint="https://test.runpod.net",
            persistent=True,
        )
        assert client.persistent
        client.close()
        assert client.session.is_closed

    def test_context_manager_non_persistent(self):
        with RunPodClient(api_key="test-key", endpoint="https://test.runpod.net") as client:
            pass
        assert client.session.is_closed

    def test_context_manager_persistent(self):
        with RunPodClient(
            api_key="test-key",
            endpoint="https://test.runpod.net",
            persistent=True,
        ) as client:
            pass
        assert not client.session.is_closed

    @patch("ivrit.client.requests.Session.request")
    def test_run_pod_with_wait(self, mock_request):
        mock_request.side_effect = [
            MagicMock(
                status_code=200,
                json=lambda: {"id": "req-123", "status": "IN_PROGRESS"},
                raise_for_status=lambda: None,
            ),
            MagicMock(
                status_code=200,
                json=lambda: {
                    "status": "COMPLETED",
                    "output": {
                        "choices": [
                            {"message": {"content": "Hello!"}}
                        ]
                    },
                },
                raise_for_status=lambda: None,
            ),
        ]

        client = RunPodClient(
            api_key="test-key",
            endpoint="https://test.runpod.net",
            poll_interval=0.01,
            timeout=1.0,
        )
        result = client.run_pod("pod-1", {"message": "hi"})
        assert result["output"]["choices"][0]["message"]["content"] == "Hello!"
        client.close()


class TestLLMProvider:
    def test_abstract_base_class(self):
        with pytest.raises(TypeError):
            LLMProvider()

    def test_runpod_llm_provider(self):
        mock_client = MagicMock(spec=RunPodClient)
        mock_client.run_pod.return_value = {
            "output": {
                "choices": [{"message": {"content": "Hi there!"}}]
            }
        }

        provider = RunPodLLMProvider(client=mock_client, pod_id="pod-1")
        result = provider.chat([{"role": "user", "content": "Hi"}])
        assert result == "Hi there!"

    def test_create_llm_provider_runpod(self):
        mock_client = MagicMock(spec=RunPodClient)
        mock_client.run_pod.return_value = {
            "output": {
                "choices": [{"message": {"content": "Hello"}}]
            }
        }
        provider = create_llm_provider("runpod", client=mock_client, pod_id="pod-1")
        assert isinstance(provider, RunPodLLMProvider)
        assert provider.chat([{"role": "user", "content": "Hi"}]) == "Hello"

    def test_create_llm_provider_unknown(self):
        with pytest.raises(ValueError, match="Unknown provider type"):
            create_llm_provider("unknown")
