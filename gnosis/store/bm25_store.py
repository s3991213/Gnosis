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

import pickle
import re
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
        self._store_path = store_path
        # Parallel lists — index i in _corpus corresponds to index i in _chunks
        self._corpus: list[list[str]] = []
        self._chunks: list[dict] = []
        self._index = None
        self._load()

    def add(self, chunks: list[Chunk]) -> None:
        """Tokenise and add *chunks* to the index, then persist."""
        if not chunks:
            return

        for chunk in chunks:
            self._corpus.append(self._tokenize(chunk.text))
            self._chunks.append(
                {
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "metadata": {**chunk.metadata, "parent_id": chunk.parent_id or ""},
                }
            )

        self._rebuild_index()
        self._save()

    def search(self, query: str, n_results: int = 10) -> list[dict]:
        """
        Return up to *n_results* chunks ranked by BM25 score.

        Each result dict contains ``id``, ``text``, ``score``, and ``metadata``.
        """
        if self._index is None or not self._chunks:
            return []

        tokens = self._tokenize(query)
        scores = self._index.get_scores(tokens)

        # Pair each score with its chunk and sort descending
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:n_results]

        results = []
        for idx, score in ranked:
            if score <= 0:
                continue
            chunk = self._chunks[idx]
            results.append(
                {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "score": float(score),
                    "metadata": {**chunk["metadata"], "source": chunk["source"]},
                }
            )
        return results

    def _tokenize(self, text: str) -> list[str]:
        """Lower-case, split on whitespace and punctuation."""
        return re.findall(r"\b\w+\b", text.lower())

    def _rebuild_index(self) -> None:
        """Rebuild the BM25Okapi index from the current corpus."""
        from rank_bm25 import BM25Okapi

        self._index = BM25Okapi(self._corpus)

    def _save(self) -> None:
        """Pickle the current index and corpus to disk."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        with self._store_path.open("wb") as f:
            pickle.dump(
                {"corpus": self._corpus, "chunks": self._chunks},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def _load(self) -> None:
        """Restore index and corpus from the pickle file if it exists."""
        if not self._store_path.exists():
            return
        with self._store_path.open("rb") as f:
            data = pickle.load(f)
        self._corpus = data.get("corpus", [])
        self._chunks = data.get("chunks", [])
        if self._corpus:
            self._rebuild_index()
