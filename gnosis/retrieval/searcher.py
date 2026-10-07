"""
Hybrid searcher for Gnosis using Reciprocal Rank Fusion (RRF).

Runs a vector query and a Postgres full-text keyword query against the
Postgres pgvector store, then merges the two ranked lists with
RRF to produce a single fused ranking. Both legs live in Postgres, so search
needs only ``DATABASE_URL`` — no separate local index.
Child chunks are used for retrieval; their parent chunks are fetched and
substituted so the LLM receives full-context passages. Children that share a
parent collapse into a single result carrying the best child's score.

RRF formula for a document *d* across rank lists *R*::

    RRF(d) = Σ  1 / (k + rank(d, R))

where ``k = 60`` (standard constant that dampens the effect of very high ranks).
"""

from __future__ import annotations

from gnosis.store.pgvector_store import PgVectorStore


class HybridSearcher:
    """
    Merge vector and keyword search results with Reciprocal Rank Fusion.

    Parameters
    ----------
    vector_store:
        Initialised PgVectorStore instance, used for both the vector and
        keyword legs of the search.
    embedder:
        Object with an ``embed_one(text) -> list[float]`` method.
    k:
        RRF constant (default 60).
    n_candidates:
        Number of candidates to fetch from each backend before fusion.
    """

    def __init__(
        self,
        vector_store: PgVectorStore,
        embedder,
        k: int = 60,
        n_candidates: int = 20,
    ) -> None:
        self._vector = vector_store
        self._embedder = embedder
        self._k = k
        self._n_candidates = n_candidates

    def search(
        self,
        query: str,
        n_results: int = 10,
        query_embedding: list[float] | None = None,
    ) -> list[dict]:
        """
        Run hybrid search for *query* and return *n_results* fused results.

        *query_embedding* overrides the vector used for the semantic leg (e.g.
        a HyDE embedding); the keyword leg always uses the raw *query* text.
        Each result dict contains ``id``, ``text``, ``score``, and ``metadata``.
        """
        if query_embedding is None:
            query_embedding = self._embedder.embed_one(query)

        # Search only among child chunks for precision
        vector_results = self._vector.query(
            query_embedding,
            n_results=self._n_candidates,
            where={"chunk_type": "child"},
        )
        keyword_results = self._vector.keyword_search(
            query,
            n_results=self._n_candidates,
            where={"chunk_type": "child"},
        )

        fused = self._rrf(vector_results, keyword_results)
        return self._expand_to_parents(fused)[:n_results]

    def _expand_to_parents(self, results: list[dict]) -> list[dict]:
        """
        Swap child text for its parent's text so the LLM gets a full passage.

        *results* must be sorted best-first; when several children share a
        parent, only the first (best-scoring) one is kept.
        """
        parent_ids = list(
            {r["metadata"]["parent_id"] for r in results if r["metadata"].get("parent_id")}
        )
        parents = {p["id"]: p for p in self._vector.get(parent_ids)}

        expanded: list[dict] = []
        seen_parents: set[str] = set()
        for r in results:
            pid = r["metadata"].get("parent_id")
            if pid in seen_parents:
                continue
            if pid in parents:
                seen_parents.add(pid)
                r = {**r, "text": parents[pid]["text"]}
            expanded.append(r)
        return expanded

    def _rrf(
        self,
        vector_results: list[dict],
        keyword_results: list[dict],
    ) -> list[dict]:
        """Fuse two ranked lists with RRF, best first."""
        scores: dict[str, float] = {}
        docs: dict[str, dict] = {}

        for rank, r in enumerate(vector_results, start=1):
            id_ = r["id"]
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (self._k + rank)
            docs[id_] = r

        for rank, r in enumerate(keyword_results, start=1):
            id_ = r["id"]
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (self._k + rank)
            if id_ not in docs:
                docs[id_] = r

        ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [{**docs[id_], "score": scores[id_]} for id_ in ranked_ids]
