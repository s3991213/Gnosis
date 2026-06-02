"""
HyDE (Hypothetical Document Embeddings) query expansion for Gnosis.

Instead of embedding the user's short query directly, HyDE asks the local
LLM to generate a short *hypothetical answer* to the query, then embeds
that answer.  Because the hypothetical answer shares vocabulary and style
with real documents, it tends to produce a richer search vector than the
original query.

Reference: Gao et al., 2022 — "Precise Zero-Shot Dense Retrieval without
Relevance Labels" (arXiv:2212.10496).
"""

from __future__ import annotations


class HyDE:
    """
    Generate a hypothetical answer embedding for a query.

    Parameters
    ----------
    llm_client:
        Object with a ``complete(prompt: str) -> str`` method (see ``llm/client.py``).
    embedder:
        Object with an ``embed_one(text: str) -> list[float]`` method.
    """

    PROMPT_TEMPLATE = (
        "Write a short, factual passage that directly answers the following question. "
        "Do not mention the question itself. Write only the passage.\n\n"
        "Question: {query}\n\nPassage:"
    )

    def __init__(self, llm_client, embedder) -> None:
        ...

    def expand(self, query: str) -> list[float]:
        """
        Generate a hypothetical answer for *query* and return its embedding.

        Falls back to embedding the raw query if the LLM call fails.
        """
        ...

    def _build_prompt(self, query: str) -> str:
        """Fill in the prompt template with *query*."""
        ...
