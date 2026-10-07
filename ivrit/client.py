import os
import time
import requests
from typing import Optional, Dict, Any, List, Callable


class RunPodClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        persistent: bool = False,
        poll_interval: float = 0.5,
        timeout: float = 300.0,
    ):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self.endpoint = endpoint or os.environ.get("RUNPOD_ENDPOINT")
        self.persistent = persistent
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self.session = requests.Session()

    def _request(
        self,
        method: str,
        path: str,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> requests.Response:
        url = f"{self.endpoint}/{path}"
        kwargs.setdefault("timeout", timeout or self.timeout)
        kwargs.setdefault("headers", self.headers)
        return self.session.request(method, url, **kwargs)

    def run_pod(
        self,
        pod_id: str,
        input_data: Dict[str, Any],
        async_mode: bool = False,
    ) -> Dict[str, Any]:
        payload = {"input": input_data}
        if async_mode:
            payload["async"] = True

        response = self._request("POST", f"run/{pod_id}", json=payload)
        response.raise_for_status()
        data = response.json()

        if not async_mode and data.get("id"):
            result = self._wait_for_result(pod_id, data["id"])
            return result

        return data

    def _wait_for_result(self, pod_id: str, request_id: str) -> Dict[str, Any]:
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            status_resp = self._request("GET", f"status/{pod_id}?id={request_id}")
            status_resp.raise_for_status()
            result = status_resp.json()

            if result.get("status") == "COMPLETED":
                return result.get("output", result)
            if result.get("status") == "ERROR":
                raise RuntimeError(f"Pod request failed: {result.get('error')}")

            time.sleep(self.poll_interval)

        raise TimeoutError(f"Request timed out after {self.timeout}s")

    def get_pod_status(self, pod_id: str) -> Dict[str, Any]:
        response = self._request("GET", f"status/{pod_id}")
        response.raise_for_status()
        return response.json()

    def stop_pod(self, pod_id: str) -> Dict[str, Any]:
        response = self._request("POST", f"stop/{pod_id}")
        response.raise_for_status()
        return response.json()

    def start_pod(self, pod_id: str) -> Dict[str, Any]:
        response = self._request("POST", f"start/{pod_id}")
        response.raise_for_status()
        return response.json()

    def delete_pod(self, pod_id: str) -> Dict[str, Any]:
        response = self._request("DELETE", f"pod/{pod_id}")
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.persistent:
            self.close()
        return False


class LLMProvider:
    """Abstract base class for LLM providers.

    Subclass this to add support for new LLM services.
    """

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        raise NotImplementedError


class RunPodLLMProvider(LLMProvider):
    """RunPod-based LLM provider."""

    def __init__(
        self,
        client: RunPodClient,
        pod_id: str,
        model: str = "default",
    ):
        self.client = client
        self.pod_id = pod_id
        self.model = model

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        result = self.client.run_pod(
            self.pod_id,
            {
                "messages": messages,
                "model": self.model,
                **kwargs,
            },
        )
        return _extract_chat_content(result)


def _extract_chat_content(result: Dict[str, Any]) -> str:
    try:
        output = result.get("output", {})
        choices = output.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
    except (AttributeError, TypeError, IndexError):
        pass
    return str(result)


def create_llm_provider(provider_type: str, **kwargs) -> LLMProvider:
    if provider_type == "runpod":
        return RunPodLLMProvider(**kwargs)
    raise ValueError(f"Unknown provider type: {provider_type}")
