"""
BM25 keyword index for Gnosis.

Wraps ``rank-bm25`` to provide fast, exact-match keyword retrieval as the
second leg of the hybrid search pipeline.  The index is serialised to disk
with ``pickle`` so it survives restarts without re-indexing.

BM25 complements the dense vector index by scoring well on rare terms,
proper nouns, code identifiers, and other strings that bi-encoder embeddings
may under-represent.
"""

from __future__ import annotations

from pathlib import Path

from gnosis.ingestion.chunker import Chunk


class BM25Store:
    """
    Persisted BM25 index over all ingested chunks.

    Parameters
    ----------
    store_path:
        Path to the pickle file used for persistence.
    """

    def __init__(self, store_path: Path) -> None:
        ...

    def add(self, chunks: list[Chunk]) -> None:
        """Tokenise and add *chunks* to the index, then persist."""
        ...

    def search(self, query: str, n_results: int = 10) -> list[dict]:
        """
        Return up to *n_results* chunks ranked by BM25 score.

        Each result dict contains ``id``, ``text``, ``score``, and ``metadata``.
        """
        ...

    def _tokenize(self, text: str) -> list[str]:
        """Lower-case, split on whitespace and punctuation."""
        ...

    def _save(self) -> None:
        """Pickle the current index and corpus to disk."""
        ...

    def _load(self) -> None:
        """Restore index and corpus from the pickle file if it exists."""
        ...
