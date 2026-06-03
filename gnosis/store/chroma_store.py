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
        When *None* embeddings must be supplied explicitly to ``upsert``.
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "gnosis",
        embedding_function=None,
    ) -> None:
        import chromadb

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_function,
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add or update *chunks* in the collection using pre-computed *embeddings*."""
        if not chunks:
            return

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    **c.metadata,
                    "source": c.source,
                    "parent_id": c.parent_id or "",
                }
                for c in chunks
            ],
        )

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
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self._collection.count() or 1),
            "include": ["documents", "distances", "metadatas"],
        }
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)

        results = []
        for id_, doc, dist, meta in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["distances"][0],
            raw["metadatas"][0],
        ):
            results.append({"id": id_, "text": doc, "distance": dist, "metadata": meta})
        return results

    def get(self, chunk_ids: list[str]) -> list[dict]:
        """Fetch chunks by exact ID."""
        if not chunk_ids:
            return []
        raw = self._collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas"],
        )
        return [
            {"id": id_, "text": doc, "metadata": meta}
            for id_, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])
        ]

    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks with the given IDs from the collection."""
        if chunk_ids:
            self._collection.delete(ids=chunk_ids)

    def count(self) -> int:
        """Return the total number of chunks in the collection."""
        return self._collection.count()
