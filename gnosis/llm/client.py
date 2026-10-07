"""
LLM client adapter for Gnosis.

Provides a unified interface for both Ollama (``http://localhost:11434``)
and LM Studio (``http://localhost:1234``) which expose an OpenAI-compatible
``/v1/chat/completions`` endpoint.

The client is intentionally minimal: it only needs to generate short text
completions for HyDE query expansion and for the ``summarize_topic`` MCP
tool.  Streaming is not required.
"""

from __future__ import annotations


class LLMClient:
    """
    HTTP client for a locally-running OpenAI-compatible LLM server.

    Parameters
    ----------
    base_url:
        Root URL of the inference server, e.g. ``http://localhost:11434``.
    model:
        Model identifier to pass in the request body.
    timeout:
        HTTP request timeout in seconds.
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "qwen3:14b"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = 300,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """
        Send *prompt* as a user message and return the assistant's reply.

        Raises ``RuntimeError`` on HTTP or JSON errors.
        """
        return self.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        """
        Send a list of *messages* (OpenAI chat format) and return the reply.

        Raises ``RuntimeError`` on HTTP or JSON errors.
        """
        import httpx

        try:
            resp = httpx.post(
                f"{self._base_url}/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected LLM response format: {exc}") from exc

    def chat_stream(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ):
        """
        Like :meth:`chat`, but yield the reply in pieces as the model writes it.

        Raises ``RuntimeError`` on HTTP or JSON errors.
        """
        import json

        import httpx

        try:
            with httpx.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                },
                timeout=httpx.Timeout(30, read=self._timeout),
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    choices = json.loads(payload).get("choices") or [{}]
                    delta = choices[0].get("delta", {}).get("content")
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        except (json.JSONDecodeError, AttributeError) as exc:
            raise RuntimeError(f"Unexpected LLM response format: {exc}") from exc

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    def list_models(self) -> list[str]:
        """Names of the models the server has downloaded."""
        import httpx

        resp = httpx.get(f"{self._base_url}/v1/models", timeout=5)
        resp.raise_for_status()
        return sorted(m["id"] for m in resp.json().get("data", []))

    def pull(self, model: str):
        """
        Download *model* (Ollama only), yielding ``(status, completed_bytes, total_bytes)``.

        Totals are summed across the model's layers, so they grow as layers appear.
        Raises ``RuntimeError`` if the server can't pull models or the pull fails.
        """
        import json

        import httpx

        layers: dict[str, tuple[int, int]] = {}
        try:
            with httpx.stream(
                "POST",
                f"{self._base_url}/api/pull",
                json={"model": model, "stream": True},
                timeout=httpx.Timeout(30, read=None),
            ) as resp:
                if resp.status_code == 404:
                    raise RuntimeError(
                        "This LLM server can't download models (only Ollama can). "
                        "Download it in your LLM app instead."
                    )
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if "error" in event:
                        raise RuntimeError(event["error"])
                    if event.get("digest") and event.get("total"):
                        layers[event["digest"]] = (event.get("completed", 0), event["total"])
                    done = sum(c for c, _ in layers.values())
                    total = sum(t for _, t in layers.values())
                    yield event.get("status", ""), done, total
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Download failed: {exc}") from exc

    def is_available(self) -> bool:
        """Return ``True`` if the server responds to a lightweight health-check."""
        import httpx

        try:
            httpx.get(f"{self._base_url}/v1/models", timeout=3).raise_for_status()
            return True
        except Exception:
            return False
