"""
MCP tool definitions for Gnosis.

Each function decorated with ``@mcp.tool`` becomes a callable tool that an
MCP-compatible LLM client (Claude Desktop, OpenWebUI, etc.) can invoke.

Tools
-----
search_memory
    Semantic + keyword hybrid search over ingested documents.
add_note
    Ingest a short note passed as a string directly through the MCP interface.
summarize_topic
    Retrieve relevant chunks and ask the local LLM to synthesise a summary.
ask_question
    Answer a question with the local LLM, citing the passages it used.
find_connections
    Surface conceptually related chunks that a user may have forgotten.
"""

from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("Gnosis")

_SUMMARY_PROMPT = """You are a research assistant. Based on the following passages from a personal
knowledge base, write a concise, factual summary about: {topic}

Passages:
{passages}

Summary:"""


@mcp.tool
async def search_memory(query: str, n_results: int = 5) -> list[dict]:
    """
    Search the Gnosis knowledge base for content relevant to *query*.

    Parameters
    ----------
    query:
        Natural-language search query.
    n_results:
        Maximum number of results to return (default 5).

    Returns a list of result dicts, each with ``text``, ``source``, and ``score``.
    """
    from gnosis.config import get_reranker, get_searcher

    searcher = get_searcher()
    candidates = searcher.search(query, n_results=n_results * 4)

    if not candidates:
        return []

    reranked = get_reranker().rerank(query, candidates, top_n=n_results)
    return [
        {
            "text": r["text"],
            "source": r.get("metadata", {}).get("source", r.get("id", "")),
            "score": r.get("rerank_score", r.get("score", 0.0)),
        }
        for r in reranked
    ]


@mcp.tool
async def add_note(content: str, title: str = "") -> dict:
    """
    Ingest *content* as a new note into the knowledge base.

    Parameters
    ----------
    content:
        The text of the note to ingest.
    title:
        Optional title used as the ``source`` metadata field.

    Returns a dict with ``chunk_count`` and ``note_id``.
    """
    import hashlib

    from gnosis.config import get_embedder, get_vector_store
    from gnosis.ingestion.chunker import chunk_documents
    from gnosis.ingestion.loader import Document

    note_id = hashlib.sha256(content.encode()).hexdigest()[:12]
    source = title or f"note:{note_id}"

    doc = Document(content=content, source=source, metadata={"type": "note"})
    chunks = chunk_documents([doc])

    vector_store = get_vector_store()
    vector_store.save_note(source, content)
    existing = vector_store.existing_ids([c.chunk_id for c in chunks])
    new_chunks = [c for c in chunks if c.chunk_id not in existing]

    if not new_chunks:
        return {"chunk_count": 0, "note_id": note_id, "status": "duplicate"}

    embedder = get_embedder()
    embeddings = embedder.embed([c.text for c in new_chunks])

    vector_store.upsert(new_chunks, embeddings)

    return {"chunk_count": len(new_chunks), "note_id": note_id, "status": "added"}


@mcp.tool
async def summarize_topic(topic: str, n_sources: int = 8) -> str:
    """
    Retrieve relevant chunks about *topic* and synthesise a summary.

    Parameters
    ----------
    topic:
        Subject to summarise.
    n_sources:
        Number of source chunks to retrieve before summarising.

    Returns the LLM-generated summary as a plain string.
    """
    return summarize(topic, n_sources)


def summarize(topic: str, n_sources: int = 8) -> str:
    """Shared by the ``summarize_topic`` tool and the CLI menu."""
    from gnosis.config import get_llm_client, get_reranker, get_searcher

    searcher = get_searcher()
    candidates = searcher.search(topic, n_results=n_sources * 3)

    if not candidates:
        return f"No information about '{topic}' found in the knowledge base."

    reranked = get_reranker().rerank(topic, candidates, top_n=n_sources)
    passages = "\n\n---\n\n".join(r["text"] for r in reranked)

    llm = get_llm_client()
    if not llm.is_available():
        return (
            f"Found {len(reranked)} relevant passage(s) but the local LLM is not running. "
            f"Start Ollama or LM Studio to enable summaries.\n\n"
            + passages
        )

    prompt = _SUMMARY_PROMPT.format(topic=topic, passages=passages)
    return llm.complete(prompt, max_tokens=1024)


@mcp.tool
async def ask_question(question: str, n_sources: int = 5) -> dict:
    """
    Answer *question* from the knowledge base using the local LLM.

    The answer cites passages inline as ``[1]``, ``[2]``; ``sources`` lists
    every passage the LLM was given, numbered to match. If nothing relevant is
    found, ``answer`` says so and ``sources`` is empty.

    Returns a dict with ``answer`` and ``sources`` (each ``n``, ``source``, ``text``).
    """
    from gnosis.config import get_llm_client
    from gnosis.retrieval.qa import answer, find_sources, source_label

    sources = find_sources(question, n=n_sources)
    listed = [
        {"n": i, "source": source_label(s), "text": s["text"]}
        for i, s in enumerate(sources, start=1)
    ]
    if not sources:
        return {"answer": "Nothing in the knowledge base seems to answer that.", "sources": []}
    if not get_llm_client().is_available():
        return {
            "answer": "The local LLM is not running; the relevant passages are in sources.",
            "sources": listed,
        }
    return {"answer": answer(question, sources), "sources": listed}


@mcp.tool
async def find_connections(query: str, n_results: int = 5) -> list[dict]:
    """
    Find chunks conceptually related to *query* that may not be obvious.

    Unlike ``search_memory``, this is pure embedding similarity: no keyword
    matching and no reranking, both of which favour passages that literally
    restate the query. Results are short child excerpts rather than full
    parent passages.

    Returns a list of result dicts with ``text``, ``source``, and ``score``.
    """
    from gnosis.config import get_embedder, get_vector_store

    embedder = get_embedder()
    query_emb = embedder.embed_one(query)

    results = get_vector_store().query(
        query_emb,
        n_results=n_results,
        where={"chunk_type": "child"},
    )

    return [
        {
            "text": r["text"],
            "source": r.get("metadata", {}).get("source", r.get("id", "")),
            "score": 1.0 - r.get("distance", 1.0),
        }
        for r in results
    ]
