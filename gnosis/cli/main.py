"""
Gnosis CLI — command-line interface built with Typer.

Commands
--------
gnosis add <path-or-url>
    Ingest a file, directory, or URL into the knowledge base.
gnosis search <query>
    Run a hybrid search and print the top results.
gnosis ask <question>
    Answer a question from the knowledge base, citing the passages used.
gnosis serve
    Start the MCP server (stdio or SSE transport).

Run ``gnosis --help`` for full usage information.
"""

from __future__ import annotations

import time

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from gnosis.cli.formatting import THEME

app = typer.Typer(
    name="gnosis",
    help="A local, private second brain powered by RAG and MCP.",
    add_completion=False,
    invoke_without_command=True,
)

_console = Console(theme=THEME)
_err = Console(stderr=True)


@app.callback()
def main(ctx: typer.Context) -> None:
    """Run with no command to open the interactive menu."""
    if ctx.invoked_subcommand is None:
        from gnosis.cli.menu import run

        run()


_EMBED_BATCH = 128
_STORE_BATCH = 200


def _progress() -> Progress:
    """The ingest loading screen: one row per stage, ticked off as it finishes."""
    return Progress(
        SpinnerColumn(finished_text="[green]✓[/green]"),
        TextColumn("{task.description:<16}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("[dim]·[/dim]"),
        TimeRemainingColumn(),
        console=_console,
        expand=True,
    )


def ingest_source(source: str) -> bool:
    """Load *source* (file, folder or URL) and ingest it with a loading screen.

    Returns False if nothing could be loaded.
    """
    from gnosis.ingestion.loader import expand, load

    try:
        items = expand(source)
    except (FileNotFoundError, ValueError) as exc:
        _err.print(f"[red]Error:[/red] {exc}")
        return False
    if not items:
        _console.print("[yellow]No supported files found there (.md, .txt, .pdf).[/yellow]")
        return False

    _console.print(f"[bold]Importing[/bold] {source}\n")
    started = time.monotonic()
    docs, failed = [], []
    with _progress() as progress:
        reading = progress.add_task("Reading files", total=len(items))
        for item in items:
            try:
                docs.extend(load(item))
            except Exception as exc:  # one bad file shouldn't stop the rest
                failed.append((item, exc))
            progress.advance(reading)

        if not docs:
            progress.stop()
            _report_failures(failed)
            _console.print("[yellow]No text could be read from that source.[/yellow]")
            return False
        result = _ingest(docs, progress)

    _report_failures(failed)
    _summary(result, n_files=len(items) - len(failed), n_docs=len(docs), started=started)
    return True


def ingest_documents(docs: list) -> None:
    """Ingest already-loaded *docs* (e.g. a note typed in the menu) with a loading screen."""
    started = time.monotonic()
    with _progress() as progress:
        result = _ingest(docs, progress)
    _summary(result, n_files=None, n_docs=len(docs), started=started)


def _ingest(docs: list, progress: Progress) -> dict:
    """Chunk, embed and store *docs*, skipping unchanged chunks and pruning stale ones."""
    from gnosis.config import get_embedder, get_vector_store
    from gnosis.ingestion.chunker import chunk_documents

    chunking = progress.add_task("Splitting text", total=len(docs))
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_documents([doc]))
        progress.advance(chunking)

    vector_store = get_vector_store()
    existing = vector_store.existing_ids([c.chunk_id for c in all_chunks])
    new_chunks = [c for c in all_chunks if c.chunk_id not in existing]

    # Search only matches child chunks; parents are fetched by ID afterwards,
    # so embedding them (the longest texts) would be wasted work.
    parents = [c for c in new_chunks if not c.parent_id]
    to_embed = [c for c in new_chunks if c.parent_id]

    # Parents go in first so a saved child never points at a missing parent.
    for i in range(0, len(parents), _STORE_BATCH):
        batch = parents[i : i + _STORE_BATCH]
        vector_store.upsert(batch, [None] * len(batch))

    # Save each batch as soon as it's embedded: if the run is interrupted,
    # the next one skips everything already stored instead of starting over.
    embedding = progress.add_task("Embedding", total=len(to_embed))
    if to_embed:
        embedder = get_embedder()
        for i in range(0, len(to_embed), _EMBED_BATCH):
            batch = to_embed[i : i + _EMBED_BATCH]
            vector_store.upsert(batch, embedder.embed([c.text for c in batch]))
            progress.advance(embedding, len(batch))

    # Drop chunks left over from earlier versions of each re-ingested source.
    # Runs after the upsert so a failure mid-ingest never loses current content.
    ids_by_source: dict[str, list[str]] = {}
    for c in all_chunks:
        ids_by_source.setdefault(c.source, []).append(c.chunk_id)
    n_removed = sum(vector_store.delete_stale(src, ids) for src, ids in ids_by_source.items())

    return {"added": len(new_chunks), "skipped": len(existing), "removed": n_removed}


def _report_failures(failed: list) -> None:
    for item, exc in failed:
        _err.print(f"[red]Couldn't read[/red] {item}: {exc}")


def _summary(result: dict, n_files: int | None, n_docs: int, started: float) -> None:
    if not result["added"] and not result["removed"]:
        _console.print("[yellow]Nothing new to ingest.[/yellow] Knowledge base is up to date.")
        return

    elapsed = time.monotonic() - started
    took = f"{elapsed:.1f}s" if elapsed < 60 else f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
    parts = []
    if n_files is not None:
        parts.append(f"[bold]{n_files}[/bold] file{'s' if n_files != 1 else ''}")
        if n_docs != n_files:  # PDFs load one document per page
            parts.append(f"[bold]{n_docs}[/bold] pages/sections")
    parts.append(f"[bold]{result['added']}[/bold] new chunks")
    if result["skipped"]:
        parts.append(f"{result['skipped']} unchanged")
    if result["removed"]:
        parts.append(f"{result['removed']} outdated removed")
    parts.append(f"took {took}")
    _console.print("\n[green]Done.[/green] " + "  ·  ".join(parts))


@app.command()
def add(
    source: str = typer.Argument(..., help="File path, directory, or URL to ingest."),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recurse into subdirectories."),
) -> None:
    """Ingest a file, directory, or URL into the Gnosis knowledge base."""
    if not ingest_source(source):
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language search query."),
    n: int = typer.Option(5, "--results", "-n", help="Number of results to show."),
    rerank: bool = typer.Option(True, help="Apply cross-encoder reranking."),
    hyde: bool = typer.Option(False, "--hyde", help="Use HyDE query expansion (requires local LLM)."),
) -> None:
    """Search the Gnosis knowledge base and print the top results."""
    run_search(query, n=n, rerank=rerank, hyde=hyde)


def run_search(query: str, n: int = 5, rerank: bool = True, hyde: bool = False) -> None:
    """Search and print results as panels."""
    results = find_results(query, n=n, rerank=rerank, hyde=hyde)
    if not results:
        _console.print("[yellow]No results found.[/yellow]")
        return

    _console.print()
    for i, result in enumerate(results, start=1):
        print_result(i, result)


def find_results(query: str, n: int = 5, rerank: bool = True, hyde: bool = False) -> list[dict]:
    """Hybrid search (optionally HyDE-expanded and reranked) with progress spinners."""
    from gnosis.config import get_embedder, get_llm_client, get_reranker, get_searcher

    searcher = get_searcher()
    query_embedding = None

    if hyde:
        llm = get_llm_client()
        if not llm.is_available():
            _err.print(
                "[yellow]Warning:[/yellow] Local LLM not reachable — falling back to plain query."
            )
        else:
            from gnosis.retrieval.hyde import HyDE

            with _console.status("[bold]Expanding query with HyDE...[/bold]"):
                query_embedding = HyDE(llm, get_embedder()).expand(query)

    with _console.status("[bold]Searching...[/bold]"):
        # n_candidates scales with requested results so reranker has enough to work with
        n_candidates = max(n * 4, 20)
        results = searcher.search(query, n_results=n_candidates, query_embedding=query_embedding)

    if not results:
        return []

    if rerank:
        with _console.status("[bold]Reranking...[/bold]"):
            return get_reranker().rerank(query, results, top_n=n)
    return results[:n]


def result_source(result: dict) -> str:
    return result.get("metadata", {}).get("source") or result.get("id", "unknown")


def print_result(
    i: int, result: dict, max_chars: int | None = 600, markdown: bool = False
) -> None:
    """Render one search result as a panel (optionally as formatted Markdown)."""
    from gnosis.cli.formatting import NoteMarkdown

    score = result.get("rerank_score") or result.get("score", 0.0)

    text = result["text"].strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "…"

    header = Text()
    header.append(f"#{i} ", style="bold cyan")
    header.append(result_source(result), style="dim")
    header.append(f"  score: {score:.4f}", style="bold green")

    _console.print(
        Panel(
            NoteMarkdown(text) if markdown else text,
            title=header,
            title_align="left",
            border_style="dim",
        )
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer from your knowledge base."),
    n: int = typer.Option(5, "--sources", "-n", help="Number of passages to answer from."),
) -> None:
    """Answer a question using your notes and documents, with citations."""
    if run_ask(question, n=n) is None:
        raise typer.Exit(1)


def run_ask(question: str, history: list | None = None, n: int = 5):
    """Answer *question* (streamed as it's written), then list the cited sources.

    Returns the :class:`~gnosis.retrieval.qa.Turn` to pass as history for a
    follow-up, or None when nothing was answered.
    """
    from rich.live import Live

    from gnosis.cli.formatting import NoteMarkdown
    from gnosis.config import get_llm_client
    from gnosis.retrieval.qa import Turn, cited, find_sources, source_label, stream_answer

    history = history or []
    with _console.status("[bold]Searching your notes...[/bold]"):
        sources = find_sources(question, history, n=n)
    if not sources:
        _console.print("[yellow]Nothing in your knowledge base seems to answer that.[/yellow]")
        return None

    if not get_llm_client().is_available():
        _console.print(
            "[yellow]The local LLM isn't running, so here are the most relevant passages instead."
            "[/yellow]\n"
        )
        for i, s in enumerate(sources, start=1):
            print_result(i, s)
        return None

    def panel(text: str) -> Panel:
        return Panel(
            NoteMarkdown(text) if text else Text("Thinking…", style="dim"),
            title=f"[bold]{question}[/bold]",
            title_align="left",
            border_style="cyan",
        )

    text = ""
    try:
        with Live(panel(""), console=_console, vertical_overflow="visible") as live:
            for text in stream_answer(question, sources, history):
                live.update(panel(text))
    except RuntimeError as exc:
        _console.print(f"[red]Error:[/red] {exc}")
        return None

    used = cited(text, sources)
    _console.print()
    _console.print("[bold]Sources[/bold]" if used else "[bold]Passages it was given[/bold]")
    for i, s in used or list(enumerate(sources, start=1)):
        _console.print(f"  [cyan]\\[{i}][/cyan] {source_label(s)}")
    return Turn(question, text)


@app.command()
def serve(
    transport: str = typer.Option("stdio", help="Transport: 'stdio' or 'sse'."),
    host: str = typer.Option("0.0.0.0", help="Bind address for SSE transport."),
    port: int = typer.Option(8000, help="Port for SSE transport."),
) -> None:
    """Start the Gnosis MCP server."""
    from gnosis.mcp_server.server import run

    run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    app()
