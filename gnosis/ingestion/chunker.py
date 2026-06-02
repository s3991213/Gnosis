"""
Document chunker for Gnosis using a parent-child strategy.

Parent chunks are large, context-rich passages (≈1 000 tokens) stored as
the text returned to the LLM.  Child chunks are small, precise excerpts
(≈200 tokens, 50 % overlap) whose embeddings are used for retrieval.
Each child carries a ``parent_id`` so the retrieval layer can fetch the
full parent on a hit.

This two-level design improves both precision (small search units) and
coherence (large context window for the LLM).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gnosis.ingestion.loader import Document


@dataclass
class Chunk:
    """A chunk of text ready for embedding and storage."""

    text: str
    source: str
    chunk_id: str
    parent_id: str | None = None
    metadata: dict = field(default_factory=dict)


class ParentChildChunker:
    """
    Split a Document into parent and child chunks.

    Parameters
    ----------
    parent_size:
        Approximate character length of each parent chunk.
    child_size:
        Approximate character length of each child chunk.
    child_overlap:
        Overlap in characters between consecutive child chunks.
    """

    def __init__(
        self,
        parent_size: int = 1500,
        child_size: int = 300,
        child_overlap: int = 50,
    ) -> None:
        ...

    def chunk(self, document: Document) -> list[Chunk]:
        """Return parent + child Chunks derived from *document*."""
        ...


def chunk_documents(documents: list[Document], **kwargs) -> list[Chunk]:
    """Convenience wrapper: chunk a list of Documents with shared settings."""
    ...
