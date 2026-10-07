"""
Cross-encoder reranker for Gnosis.

After the hybrid search returns a set of candidate chunks, the reranker
scores each (query, chunk) pair with a cross-encoder model
(``cross-encoder/ms-marco-MiniLM-L-6-v2``).  Cross-encoders apply full
self-attention over the concatenated query and document, yielding much
more accurate relevance scores than the bi-encoder used for retrieval.

The reranker is the final filter before results are handed to the LLM.
"""

from __future__ import annotations


class Reranker:
    """
    Re-score a list of candidate chunks with a cross-encoder model.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for the cross-encoder.
    top_n:
        Default number of top-scored chunks to return; can be overridden
        per call in :meth:`rerank`.
    device:
        Torch device string.  Auto-detected when *None*.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        top_n: int = 5,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device=device)
        self._top_n = top_n

    def rerank(self, query: str, candidates: list[dict], top_n: int | None = None) -> list[dict]:
        """
        Score each candidate against *query* and return the top *top_n*
        (defaulting to the value given at construction).

        *candidates* must have a ``"text"`` key.  The returned list is sorted
        by descending cross-encoder score and each dict gains a ``"rerank_score"`` key.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        results = []
        for chunk, score in ranked[: top_n or self._top_n]:
            results.append({**chunk, "rerank_score": float(score)})
        return results
