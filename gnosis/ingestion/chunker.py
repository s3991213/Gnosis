"""
Document chunker for Gnosis using a parent-child strategy.

Parent chunks are large, context-rich passages (≈1 500 chars) stored as
the text returned to the LLM.  Child chunks are small, precise excerpts
(≈300 chars, 50-char overlap) whose embeddings are used for retrieval.
Each child carries a ``parent_id`` so the retrieval layer can fetch the
full parent on a hit.

This two-level design improves both precision (small search units) and
coherence (large context window for the LLM).
"""

from __future__ import annotations

import hashlib
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


def _make_id(text: str, source: str, position: str) -> str:
    # *position* must differ between a parent and its first child: both start
    # with the same text, so the text prefix alone can't tell them apart.
    digest = hashlib.sha256(f"{source}:{position}:{text[:64]}".encode()).hexdigest()[:16]
    return digest


def _split(text: str, size: int, overlap: int) -> list[str]:
    """Split *text* into overlapping windows of roughly *size* characters."""
    if len(text) <= size:
        return [text]

    segments: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        if end >= len(text):
            segments.append(text[start:].strip())
            break
        # Try to break on a newline or space to avoid cutting mid-word. Only
        # accept a boundary past the overlap, otherwise the next window would
        # start at or before this one and the loop would never advance.
        for sep in ("\n\n", "\n", " "):
            boundary = text.rfind(sep, start + overlap + 1, end)
            if boundary != -1:
                end = boundary + len(sep)
                break
        segments.append(text[start:end].strip())
        start = end - overlap
    return [s for s in segments if s]


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
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        """Return parent + child Chunks derived from *document*."""
        all_chunks: list[Chunk] = []
        parent_texts = _split(document.content, self.parent_size, self.child_size // 2)

        for p_idx, parent_text in enumerate(parent_texts):
            parent_id = _make_id(parent_text, document.source, f"p{p_idx}")
            parent_chunk = Chunk(
                text=parent_text,
                source=document.source,
                chunk_id=parent_id,
                parent_id=None,
                metadata={**document.metadata, "chunk_type": "parent", "position": p_idx},
            )
            all_chunks.append(parent_chunk)

            child_texts = _split(parent_text, self.child_size, self.child_overlap)
            for c_idx, child_text in enumerate(child_texts):
                child_id = _make_id(child_text, document.source, f"c{p_idx}.{c_idx}")
                all_chunks.append(
                    Chunk(
                        text=child_text,
                        source=document.source,
                        chunk_id=child_id,
                        parent_id=parent_id,
                        metadata={**document.metadata, "chunk_type": "child"},
                    )
                )

        return all_chunks


def chunk_documents(documents: list[Document], **kwargs) -> list[Chunk]:
    """Convenience wrapper: chunk a list of Documents with shared settings."""
    chunker = ParentChildChunker(**kwargs)
    result: list[Chunk] = []
    for doc in documents:
        result.extend(chunker.chunk(doc))
    return result
