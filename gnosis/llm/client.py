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
        timeout: int = 60,
    ) -> None:
        ...

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """
        Send *prompt* as a user message and return the assistant's reply.

        Raises ``RuntimeError`` on HTTP or JSON errors.
        """
        ...

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
        ...

    def is_available(self) -> bool:
        """Return ``True`` if the server responds to a lightweight health-check."""
        ...
