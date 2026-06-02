"""
Content deduplicator for Gnosis.

Uses SHA-256 hashes of chunk text to detect whether a piece of content has
already been ingested.  Hashes are persisted in a simple JSON file so the
check survives restarts.

This prevents redundant re-embedding when ``gnosis add`` is run on a folder
that has not changed, and avoids polluting the vector store with duplicate
vectors when the same document is added twice.
"""

from __future__ import annotations

from pathlib import Path

from gnosis.ingestion.chunker import Chunk


class Deduplicator:
    """
    Hash-based chunk deduplicator backed by a JSON file.

    Parameters
    ----------
    store_path:
        Path to the JSON file that persists seen hashes.
    """

    def __init__(self, store_path: Path) -> None:
        ...

    def is_duplicate(self, chunk: Chunk) -> bool:
        """Return ``True`` if *chunk* has been seen before."""
        ...

    def mark_seen(self, chunk: Chunk) -> None:
        """Record the hash of *chunk* so future calls to ``is_duplicate`` return ``True``."""
        ...

    def filter(self, chunks: list[Chunk]) -> tuple[list[Chunk], int]:
        """
        Return only unseen chunks together with a count of duplicates dropped.

        Also marks every returned chunk as seen.
        """
        ...

    def _hash(self, chunk: Chunk) -> str:
        """Compute a stable SHA-256 hex digest of *chunk*'s text."""
        ...

    def _load(self) -> None:
        """Load persisted hashes from disk."""
        ...

    def _save(self) -> None:
        """Flush in-memory hashes to disk."""
        ...
