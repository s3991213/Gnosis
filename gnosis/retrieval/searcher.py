"""
Hybrid searcher for Gnosis using Reciprocal Rank Fusion (RRF).

Runs a vector query against ChromaDB and a BM25 keyword query in parallel,
then merges the two ranked lists with RRF to produce a single fused ranking.
The fused results are optionally expanded via HyDE before retrieval and
re-scored by the cross-encoder reranker after retrieval.

RRF formula for a document *d* across rank lists *R*::

    RRF(d) = Σ  1 / (k + rank(d, R))

where ``k = 60`` (standard constant that dampens the effect of very high ranks).
"""

from __future__ import annotations

from gnosis.store.bm25_store import BM25Store
from gnosis.store.chroma_store import ChromaStore


class HybridSearcher:
    """
    Merge vector and BM25 search results with Reciprocal Rank Fusion.

    Parameters
    ----------
    chroma:
        Initialised ChromaStore instance.
    bm25:
        Initialised BM25Store instance.
    embedder:
        Object with an ``embed_one(text) -> list[float]`` method.
    k:
        RRF constant (default 60).
    n_candidates:
        Number of candidates to fetch from each backend before fusion.
    """

    def __init__(
        self,
        chroma: ChromaStore,
        bm25: BM25Store,
        embedder,
        k: int = 60,
        n_candidates: int = 20,
    ) -> None:
        ...

    def search(self, query: str, n_results: int = 10) -> list[dict]:
        """
        Run hybrid search for *query* and return *n_results* fused results.

        Each result dict contains ``id``, ``text``, ``score``, and ``metadata``.
        """
        ...

    def _rrf(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        n_results: int,
    ) -> list[dict]:
        """Fuse two ranked lists with RRF and return the top *n_results*."""
        ...
