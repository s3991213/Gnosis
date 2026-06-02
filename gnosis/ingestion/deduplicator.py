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

import hashlib
import json
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
        self._store_path = store_path
        self._seen: set[str] = set()
        self._load()

    def is_duplicate(self, chunk: Chunk) -> bool:
        """Return ``True`` if *chunk* has been seen before."""
        return self._hash(chunk) in self._seen

    def mark_seen(self, chunk: Chunk) -> None:
        """Record the hash of *chunk* so future calls to ``is_duplicate`` return ``True``."""
        self._seen.add(self._hash(chunk))
        self._save()

    def filter(self, chunks: list[Chunk]) -> tuple[list[Chunk], int]:
        """
        Return only unseen chunks together with a count of duplicates dropped.

        Also marks every returned chunk as seen.
        """
        new_chunks: list[Chunk] = []
        duplicates = 0
        new_hashes: set[str] = set()

        for chunk in chunks:
            h = self._hash(chunk)
            if h in self._seen or h in new_hashes:
                duplicates += 1
            else:
                new_chunks.append(chunk)
                new_hashes.add(h)

        self._seen.update(new_hashes)
        if new_hashes:
            self._save()

        return new_chunks, duplicates

    def _hash(self, chunk: Chunk) -> str:
        """Compute a stable SHA-256 hex digest of *chunk*'s text."""
        return hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        """Load persisted hashes from disk."""
        if self._store_path.exists():
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
            self._seen = set(data.get("hashes", []))

    def _save(self) -> None:
        """Flush in-memory hashes to disk."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._store_path.write_text(
            json.dumps({"hashes": list(self._seen)}, indent=2),
            encoding="utf-8",
        )
