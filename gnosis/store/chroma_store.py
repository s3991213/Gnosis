"""
ChromaDB vector store for Gnosis.

Manages a persistent ChromaDB collection that stores chunk embeddings and
their associated metadata.  Supports upsert (add-or-update), exact-ID
lookup, and approximate nearest-neighbour search.

The store uses the same embedding function as the ingestion pipeline so
query vectors are always in the same space as document vectors.
"""

from __future__ import annotations

from pathlib import Path

from gnosis.ingestion.chunker import Chunk


class ChromaStore:
    """
    Thin wrapper around a ChromaDB persistent collection.

    Parameters
    ----------
    persist_dir:
        Directory where ChromaDB stores its files on disk.
    collection_name:
        Name of the ChromaDB collection to use or create.
    embedding_function:
        Callable that maps a list of strings to a list of float vectors.
        When *None* Chroma's default (all-MiniLM-L6-v2) is used.
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "gnosis",
        embedding_function=None,
    ) -> None:
        ...

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add or update *chunks* in the collection using pre-computed *embeddings*."""
        ...

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Return up to *n_results* chunks closest to *query_embedding*.

        Each result dict contains ``id``, ``text``, ``distance``, and ``metadata``.
        """
        ...

    def get(self, chunk_ids: list[str]) -> list[dict]:
        """Fetch chunks by exact ID."""
        ...

    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks with the given IDs from the collection."""
        ...

    def count(self) -> int:
        """Return the total number of chunks in the collection."""
        ...
