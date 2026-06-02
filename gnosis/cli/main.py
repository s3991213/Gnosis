"""
Gnosis CLI — command-line interface built with Typer.

Commands
--------
gnosis add <path-or-url>
    Ingest a file, directory, or URL into the knowledge base.
gnosis search <query>
    Run a hybrid search and print the top results.
gnosis serve
    Start the MCP server (stdio or SSE transport).

Run ``gnosis --help`` for full usage information.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="gnosis",
    help="A local, private second brain powered by RAG and MCP.",
    add_completion=False,
)


@app.command()
def add(
    source: str = typer.Argument(..., help="File path, directory, or URL to ingest."),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recurse into subdirectories."),
) -> None:
    """Ingest a file, directory, or URL into the Gnosis knowledge base."""
    ...


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language search query."),
    n: int = typer.Option(5, "--results", "-n", help="Number of results to show."),
    rerank: bool = typer.Option(True, help="Apply cross-encoder reranking."),
) -> None:
    """Search the Gnosis knowledge base and print the top results."""
    ...


@app.command()
def serve(
    transport: str = typer.Option("stdio", help="Transport: 'stdio' or 'sse'."),
    host: str = typer.Option("0.0.0.0", help="Bind address for SSE transport."),
    port: int = typer.Option(8000, help="Port for SSE transport."),
) -> None:
    """Start the Gnosis MCP server."""
    ...


if __name__ == "__main__":
    app()
