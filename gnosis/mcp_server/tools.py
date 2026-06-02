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
find_connections
    Surface conceptually related chunks that a user may have forgotten.
"""

from __future__ import annotations


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
    ...


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
    ...


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
    ...


async def find_connections(query: str, n_results: int = 5) -> list[dict]:
    """
    Find chunks conceptually related to *query* that may not be obvious.

    Uses a broader embedding search with a lower similarity threshold than
    ``search_memory`` to surface unexpected connections.

    Returns a list of result dicts with ``text``, ``source``, and ``score``.
    """
    ...
