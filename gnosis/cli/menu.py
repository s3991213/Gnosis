"""
Interactive menu for Gnosis — what ``gnosis`` runs with no command.

Arrow-key navigation (via questionary) over questions, search, browsing, writing notes,
importing files/URLs and summarising topics. Every screen starts at the top of
the window; earlier output is scrolled up rather than erased, so it stays in
the terminal's scrollback.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path

import questionary
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.markup import MarkdownLexer
from questionary import Choice, Separator, Style
from rich.console import Console
from rich.control import Control
from rich.panel import Panel
from rich.text import Text

from gnosis.cli.formatting import FORMATTING_HELP, THEME, NoteMarkdown
from gnosis.retrieval.qa import WEAK_SCORE

_console = Console(theme=THEME)

# Folder the Docker stack mounts from ./inbox on the host; offered as the
# default import path so files dropped there can be added with one keypress.
_INBOX = os.getenv("GNOSIS_INBOX", "")

_STYLE = Style(
    [
        ("qmark", "fg:ansimagenta bold"),
        ("question", "bold"),
        ("pointer", "fg:ansicyan bold"),
        ("highlighted", "fg:ansicyan bold"),
        ("selected", "fg:ansicyan"),
        ("instruction", "fg:ansibrightblack italic"),
        ("answer", "fg:ansicyan bold"),
        ("separator", "fg:ansibrightblack"),
    ]
)
_NAV_HINT = "(↑/↓ to move, Enter to choose)"
_DIM = "fg:ansibrightblack"

# Navigation choices, distinct from any note or result value.
_BACK, _NEW_SEARCH, _PREV, _NEXT, _FILTER, _CLEAR_FILTER, _EDIT, _DELETE = (
    "<back>",
    "<new-search>",
    "<prev>",
    "<next>",
    "<filter>",
    "<clear-filter>",
    "<edit>",
    "<delete>",
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _fresh_screen() -> None:
    """Start the next screen at the top of the window.

    Scrolls the previous output up out of view instead of erasing it, so it
    can still be reached by scrolling back.
    """
    _console.print("\n" * (_console.height - 1), end="")
    _console.control(Control.home())


def _select(message: str, choices: list, default=None):
    """Arrow-key select. Returns None on Ctrl+C."""
    values = [c.value for c in choices if isinstance(c, Choice)]
    return questionary.select(
        message,
        choices=choices,
        default=default if default in values else None,
        qmark="",
        instruction=_NAV_HINT,
        style=_STYLE,
    ).ask()


def _text(message: str, default: str = "") -> str | None:
    """One-line text prompt. Returns None on Ctrl+C."""
    return questionary.text(message, default=default, qmark="", style=_STYLE).ask()


def _wait(message: str = "Press any key to go back") -> None:
    _console.print()
    questionary.press_any_key_to_continue(f"{message}…", style=_STYLE).ask()


def _display_name(source: str) -> str:
    """File paths show as their filename; URLs and note titles as-is."""
    if source.startswith("note:"):
        return f"Untitled note ({source[5:11]})"
    return Path(source).name if source.startswith("/") else source


def _confirm(message: str) -> bool:
    return bool(questionary.confirm(message, default=False, qmark="", style=_STYLE).ask())


def _edit_text(default: str = "") -> str | None:
    """Multi-line editor with Markdown highlighting. Returns None if cancelled."""
    keys = KeyBindings()

    @keys.add("c-s")
    def _save(event) -> None:
        event.current_buffer.validate_and_handle()

    _console.print(f"[dim]{FORMATTING_HELP}[/dim]\n")
    return questionary.text(
        "",
        default=default,
        multiline=True,
        qmark="",
        style=_STYLE,
        instruction="Ctrl+S to save (or Esc then Enter), Ctrl+C to cancel\n",
        lexer=PygmentsLexer(MarkdownLexer),
        key_bindings=keys,
    ).ask()


def _save_note(source: str, content: str) -> None:
    """Store a note's full text and (re)index it."""
    from gnosis.cli.main import ingest_documents
    from gnosis.config import get_vector_store
    from gnosis.ingestion.loader import Document

    get_vector_store().save_note(source, content)
    ingest_documents([Document(content=content, source=source, metadata={"type": "note"})])


def _note_source(title: str, content: str) -> str:
    return title or f"note:{hashlib.sha256(content.encode()).hexdigest()[:12]}"


def _compose(title: str, body: str) -> str:
    return f"# {title}\n\n{body}" if title else body


def _split_title(source: str, content: str) -> tuple[str, str]:
    """Inverse of _compose for a stored note."""
    title = "" if source.startswith("note:") else source
    heading = f"# {title}\n\n"
    if title and content.startswith(heading):
        return title, content[len(heading) :]
    return title, content


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(width - 1, 0)] + "…"


def _snippet(text: str, width: int) -> str:
    """First meaningful line of *text*, trimmed to *width* characters."""
    lines = (ln.strip("#-*> \t") for ln in text.splitlines())
    line = re.sub(r"\s+", " ", next((ln for ln in lines if ln), ""))
    return _clip(line, width)


def _join_passages(passages: list[str]) -> str:
    """Join consecutive parent passages, dropping the text they overlap by."""
    if not passages:
        return ""
    out = passages[0]
    for nxt in passages[1:]:
        for k in range(min(len(out), len(nxt), 400), 0, -1):
            if out.endswith(nxt[:k]):
                nxt = nxt[k:]
                break
        out = f"{out}\n{nxt}"
    return out


# ── screens ──────────────────────────────────────────────────────────────────


def _header() -> Panel:
    from gnosis.config import get_llm_client, get_vector_store

    n_sources = get_vector_store().count_sources()
    llm = get_llm_client()
    llm_up = llm.is_available()

    status = Text()
    status.append(f"{n_sources} ", style="bold")
    status.append("note" if n_sources == 1 else "notes")
    status.append(" in your knowledge base   ")
    status.append("● ", style="green" if llm_up else "red")
    status.append(llm.model if llm_up else "LLM offline", style="dim")

    return Panel(
        status,
        title="[bold magenta]Gnosis[/bold magenta]",
        subtitle="[dim]your second brain[/dim]",
        border_style="magenta",
        padding=(1, 2),
    )


def _search() -> None:
    from gnosis.cli.main import find_results, print_result, result_source

    query = (_text("Search:") or "").strip()
    while query:
        _fresh_screen()
        results = find_results(query)
        strong = [r for r in results if r.get("rerank_score", 0.0) > WEAK_SCORE]
        weak_only = bool(results) and not strong
        results = strong or results[:3]

        picked = None
        while True:
            _fresh_screen()
            _console.print(f'[bold]Results for[/bold] [cyan]"{query}"[/cyan]\n')
            if not results:
                _console.print("[yellow]No results found.[/yellow]\n")
            elif weak_only:
                _console.print("[yellow]No strong matches.[/yellow] [dim]Closest notes:[/dim]\n")

            snippet_width = max(_console.width - 36, 20)
            choices: list = [
                Choice(
                    [
                        ("", f"{i}. {_clip(_display_name(result_source(r)), 24):<24}  "),
                        (_DIM, _snippet(r["text"], snippet_width)),
                    ],
                    value=i,
                )
                for i, r in enumerate(results, start=1)
            ]
            if choices:
                choices.append(Separator())
            choices += [Choice("New search", _NEW_SEARCH), Choice("Back to menu", _BACK)]

            picked = _select("", choices, default=picked)
            if picked is None or picked == _BACK:
                return
            if picked == _NEW_SEARCH:
                break

            _fresh_screen()
            print_result(picked, results[picked - 1], max_chars=None, markdown=True)
            _wait("Press any key to go back to the results")

        query = (_text("Search:") or "").strip()


def _browse() -> None:
    from gnosis.config import get_vector_store

    store = get_vector_store()
    page, name_filter, picked = 0, "", None

    while True:
        sources = sorted(store.list_sources(), key=lambda s: _display_name(s["source"]).lower())
        if name_filter:
            sources = [s for s in sources if name_filter.lower() in s["source"].lower()]

        per_page = max(_console.height - 12, 5)
        n_pages = max(math.ceil(len(sources) / per_page), 1)
        page = min(page, n_pages - 1)
        shown = sources[page * per_page : (page + 1) * per_page]

        _fresh_screen()
        title = Text()
        title.append(f"{len(sources)} ", style="bold")
        title.append("note" if len(sources) == 1 else "notes")
        if name_filter:
            title.append(f' matching "{name_filter}"', style="cyan")
        if n_pages > 1:
            first = page * per_page + 1
            title.append(
                f"   showing {first}–{first + len(shown) - 1}  ·  page {page + 1} of {n_pages}",
                style="dim",
            )
        _console.print(title, "\n")

        name_width = min(max(_console.width - 20, 20), 48)
        choices: list = [
            Choice(
                [
                    ("", f"{_clip(_display_name(s['source']), name_width):<{name_width}}  "),
                    (_DIM, s["type"]),
                ],
                s["source"],
            )
            for s in shown
        ]
        if choices:
            choices.append(Separator())
        if page > 0:
            choices.append(Choice("← Previous page", _PREV))
        if page < n_pages - 1:
            choices.append(Choice("Next page →", _NEXT))
        choices.append(Choice("Find by name…", _FILTER))
        if name_filter:
            choices.append(Choice("Show all notes", _CLEAR_FILTER))
        choices.append(Choice("Back to menu", _BACK))

        picked = _select("", choices, default=picked)
        if picked is None or picked == _BACK:
            return
        if picked == _PREV:
            page -= 1
        elif picked == _NEXT:
            page += 1
        elif picked == _FILTER:
            name_filter = (_text("Name contains:") or "").strip()
            page, picked = 0, None
        elif picked == _CLEAR_FILTER:
            name_filter, page, picked = "", 0, None
        else:
            _show_note(picked)


def _show_note(source: str) -> None:
    """Show a note in full, with Edit/Delete actions. Returns after Back or Delete."""
    from gnosis.config import get_vector_store

    store = get_vector_store()
    picked = None
    while True:
        stored = store.get_note(source)
        text = stored if stored is not None else _join_passages(store.source_passages(source))
        # Notes written before full note text was stored are still editable;
        # their text is rebuilt from chunks and stored on the first save.
        editable = stored is not None or store.source_type(source) == "note"

        _fresh_screen()
        _console.print(
            Panel(
                NoteMarkdown(text) if text else Text("(empty)", style="dim"),
                title=f"[bold]{_display_name(source)}[/bold]",
                subtitle=f"[dim]{source}[/dim]" if source != _display_name(source) else None,
                title_align="left",
                border_style="cyan",
            )
        )

        choices: list = [Choice("Back to the list", _BACK)]
        if editable:
            choices.append(Choice("Edit note", _EDIT))
        else:
            _console.print(
                "[dim]This came from a file or web page — to change it, edit the original "
                "and import it again.[/dim]"
            )
        choices.append(Choice("Delete note", _DELETE))

        picked = _select("", choices, default=picked)
        if picked is None or picked == _BACK:
            return
        if picked == _DELETE:
            if _confirm(f'Delete "{_display_name(source)}"? This can\'t be undone.'):
                store.delete_source(source)
                return
        elif picked == _EDIT:
            source = _edit_note(source, text) or source


def _edit_note(source: str, content: str) -> str | None:
    """Edit a note's title and body. Returns its (possibly new) source, or None."""
    from gnosis.config import get_vector_store

    store = get_vector_store()
    old_title, old_body = _split_title(source, content)

    _fresh_screen()
    title = _text("Title (optional):", default=old_title)
    if title is None:
        return None
    body = _edit_text(old_body)
    if body is None:
        return None

    title, body = title.strip(), body.strip()
    if (title, body) == (old_title, old_body.strip()):
        return None
    if not body:
        _console.print("[yellow]A note can't be empty — use Delete instead.[/yellow]")
        _wait()
        return None

    new_content = _compose(title, body)
    new_source = _note_source(title, new_content)
    if new_source != source and store.get_note(new_source) is not None:
        if not _confirm(f'A note called "{new_source}" already exists. Replace it?'):
            return None

    _save_note(new_source, new_content)
    if new_source != source:
        store.delete_source(source)
    return new_source


def _new_note() -> None:
    from gnosis.config import get_vector_store

    title = _text("Title (optional):")
    if title is None:
        return
    title = title.strip()
    body = _edit_text()
    if body is None:  # cancelled
        return
    body = body.strip()
    if not body:
        _console.print("[yellow]Empty note — nothing saved.[/yellow]")
        _wait()
        return

    content = _compose(title, body)
    source = _note_source(title, content)
    if title and get_vector_store().get_note(source) is not None:
        if not _confirm(f'A note called "{title}" already exists. Replace it?'):
            return

    _fresh_screen()
    _save_note(source, content)
    _console.print(
        Panel(NoteMarkdown(content), title=f"[green]Saved[/green] {source}", border_style="green")
    )
    _wait()


def _import() -> None:
    from gnosis.cli.main import ingest_source

    if _INBOX:
        _console.print("[dim]Leave as is to import everything in ./inbox.[/dim]")
    source = (_text("File, folder or URL:", default=_INBOX) or "").strip()
    if not source:
        return

    _fresh_screen()
    ingest_source(source)
    _wait()


# Shown in the model picker when not yet downloaded. Sizes are Ollama's
# default (4-bit) downloads; RAM is roughly what the model needs to run.
_RECOMMENDED_MODELS = [
    ("qwen2.5:1.5b", "1.0 GB", "fastest, basic answers"),
    ("llama3.2:3b", "2.0 GB", "good balance of speed and quality"),
    ("qwen2.5:3b", "1.9 GB", "good balance of speed and quality"),
    ("qwen2.5:7b", "4.7 GB", "best answers, slower, needs ~6 GB RAM"),
    ("llama3.1:8b", "4.9 GB", "best answers, slower, needs ~6 GB RAM"),
]


def _change_model() -> None:
    from gnosis.config import LLM_MODEL_SETTING, get_llm_client, get_vector_store

    llm = get_llm_client()
    _fresh_screen()
    _console.print("[bold]AI model[/bold] [dim]— used for summaries and HyDE search[/dim]\n")
    if not llm.is_available():
        _console.print(f"[red]Can't reach the LLM server at {llm.base_url}.[/red]")
        _console.print("[dim]With Docker, it starts with ./gn. Otherwise start Ollama.[/dim]")
        _wait()
        return

    installed = llm.list_models()
    sizes = {name: size for name, size, _ in _RECOMMENDED_MODELS}
    _console.print(f"Currently using [cyan]{llm.model}[/cyan]\n")

    choices: list = [Separator("── Downloaded ──")]
    for name in installed:
        tag = "in use" if name == llm.model else ""
        choices.append(Choice([("", f"{name:<22}"), (_DIM, tag)], name))
    available = [m for m in _RECOMMENDED_MODELS if m[0] not in installed]
    if available:
        choices.append(Separator("── Download ──"))
        for name, size, hint in available:
            choices.append(Choice([("", f"{name:<22}"), (_DIM, f"{size:>7}   {hint}")], name))
    choices += [
        Separator(),
        Choice("Other model by name…", _FILTER),
        Choice("Back to menu", _BACK),
    ]

    model = _select("", choices, default=llm.model)
    if model is None or model == _BACK:
        return
    if model == _FILTER:
        _console.print("[dim]Browse models at https://ollama.com/library[/dim]")
        model = (_text("Model name (e.g. mistral:7b):") or "").strip()
        if not model:
            return

    if model not in installed:
        size = f" ({sizes[model]})" if model in sizes else ""
        if not _confirm(f"Download {model}{size}?"):
            return
        if not _download_model(llm, model):
            _wait()
            return

    get_vector_store().set_setting(LLM_MODEL_SETTING, model)
    llm = get_llm_client()
    with _console.status(f"[bold]Loading {model} into memory…[/bold]"):
        try:
            llm.complete("Reply with OK.", max_tokens=3)
            ok = True
        except RuntimeError as exc:
            ok, error = False, exc
    if ok:
        _console.print(f"\n[green]Now using[/green] [bold]{model}[/bold].")
    else:
        _console.print(f"\n[yellow]Switched to {model}, but it didn't respond:[/yellow] {error}")
        _console.print("[dim]It may need more memory than Docker has. Pick a smaller one.[/dim]")
    _wait()


def _download_model(llm, model: str) -> bool:
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    _console.print()
    columns = (
        TextColumn("{task.description:<24}"),
        BarColumn(bar_width=None),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )
    try:
        with Progress(*columns, console=_console, expand=True) as progress:
            task = progress.add_task(f"Downloading {model}", total=None)
            total = 0
            for status, done, total in llm.pull(model):
                progress.update(task, completed=done, total=total or None)
                if not status.startswith("pulling"):
                    progress.update(task, description=status.capitalize()[:24])
            # Ollama's final events omit byte counts; show the bar as full.
            progress.update(task, completed=total, total=total or 1)
    except RuntimeError as exc:
        _console.print(f"[red]Couldn't download {model}:[/red] {exc}")
        return False
    return True


def _ask() -> None:
    from gnosis.cli.main import run_ask

    history: list = []
    question = (_text("Ask:") or "").strip()
    while question:
        _fresh_screen()
        turn = run_ask(question, history)
        if turn is not None:
            history.append(turn)
        _console.print()
        prompt = "Follow-up" if history else "Ask something else"
        question = (_text(f"{prompt} (Enter to finish):") or "").strip()


def _summarize() -> None:
    from gnosis.mcp_server.tools import summarize

    topic = (_text("Summarize what I know about:") or "").strip()
    if not topic:
        return
    _fresh_screen()
    with _console.status("[bold]Reading your notes and thinking…[/bold]"):
        try:
            answer = summarize(topic)
        except RuntimeError as exc:
            answer = f"**Error:** {exc}"
    _console.print(
        Panel(NoteMarkdown(answer), title=f"[bold]{topic}[/bold]", border_style="cyan")
    )
    _wait()


_ACTIONS = [
    ("Ask a question", "answers from your notes, with sources", _ask),
    ("Search notes", "find anything by meaning", _search),
    ("Browse all notes", "see everything you've saved", _browse),
    ("Write a new note", "type it right here", _new_note),
    ("Add a file, folder or URL", "Markdown, text, PDF, web pages", _import),
    ("Summarize a topic", "uses the local LLM", _summarize),
    ("Change AI model", "pick or download a different LLM", _change_model),
]


def run() -> None:
    """Show the menu until the user quits."""
    from gnosis.config import get_embedder, get_reranker, get_vector_store

    try:
        with _console.status("[bold magenta]Waking up your second brain…[/bold magenta]"):
            get_vector_store()
            get_embedder()
            get_reranker()
    except Exception as exc:
        _console.print(f"[red]Couldn't connect to the database:[/red] {exc}")
        _console.print("Start it with [bold]docker compose up -d db[/bold] and try again.")
        return

    choices: list = [
        Choice([("", f"{label:<27} "), (_DIM, hint)], value=label) for label, hint, _ in _ACTIONS
    ]
    choices += [Separator(), Choice("Quit", _BACK)]
    actions = {label: fn for label, _, fn in _ACTIONS}
    last = None

    while True:
        _fresh_screen()
        _console.print(_header())
        choice = _select("", choices, default=last)
        if choice is None or choice == _BACK:
            break
        last = choice
        try:
            actions[choice]()
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C / Ctrl+D inside an action backs out to the menu.
            continue
        except Exception as exc:
            # Keep the menu alive; show what went wrong instead of a crash.
            _console.print(
                Panel(
                    f"{type(exc).__name__}: {exc}",
                    title="[red]Something went wrong[/red]",
                    border_style="red",
                )
            )
            _wait("Press any key to go back to the menu")
    _console.print("[magenta]Bye![/magenta]")
