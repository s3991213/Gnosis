"""
Text embedder for Gnosis.

Wraps ``sentence-transformers`` (model: ``all-MiniLM-L6-v2``) to produce
dense 384-dimensional embeddings.  The model is loaded once at construction
time and reused across calls.

Usage::

    embedder = Embedder()
    vectors = embedder.embed(["text one", "text two"])
"""

from __future__ import annotations

import numpy as np


class Embedder:
    """
    Thin wrapper around a sentence-transformers model.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Defaults to ``all-MiniLM-L6-v2``.
    device:
        Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        Auto-detected when *None*.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per text in *texts*."""
        ...

    def embed_one(self, text: str) -> list[float]:
        """Convenience method: embed a single string."""
        ...
