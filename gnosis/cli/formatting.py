"""
Note rendering for the terminal: Markdown plus highlight and underline.

Standard Markdown has no underline or highlight, so notes may use

    ==highlighted text==     (same syntax as Obsidian)
    ++underlined text++      (or <u>…</u>, <mark>…</mark>)

Rich's Markdown renderer only styles a fixed set of inline tags, so these are
rewritten to inline HTML, and the resulting HTML tokens are turned into
open/close tokens for two extra tags that Rich then styles via the theme.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from markdown_it.token import Token
from rich.markdown import Markdown
from rich.theme import Theme

THEME = Theme({"markdown.mark": "black on yellow", "markdown.u": "underline"})

_HTML_TAGS = {"<mark>": "mark", "</mark>": "mark", "<u>": "u", "</u>": "u"}
_HIGHLIGHT = re.compile(r"==(?=\S)(.+?)(?<=\S)==")
_UNDERLINE = re.compile(r"\+\+(?=\S)(.+?)(?<=\S)\+\+")
# Code spans and fenced blocks are left untouched.
_CODE = re.compile(r"(```[\s\S]*?```|`[^`\n]*`)")

FORMATTING_HELP = (
    "# Heading   **bold**   *italic*   ++underline++   ==highlight==   ~~strike~~\n"
    "- bullet    1. numbered    `code`"
)


def _extend_syntax(markup: str) -> str:
    parts = _CODE.split(markup)
    for i in range(0, len(parts), 2):  # even indices are outside code
        part = _HIGHLIGHT.sub(r"<mark>\1</mark>", parts[i])
        parts[i] = _UNDERLINE.sub(r"<u>\1</u>", part)
    return "".join(parts)


def _walk(tokens: Iterable[Token]) -> Iterable[Token]:
    for token in tokens:
        yield token
        if token.children:
            yield from _walk(token.children)


class NoteMarkdown(Markdown):
    """Rich Markdown that also renders ==highlight== and ++underline++."""

    inlines = Markdown.inlines | {"mark", "u"}

    def __init__(self, markup: str, **kwargs) -> None:
        super().__init__(_extend_syntax(markup), **kwargs)
        for token in _walk(self.parsed):
            tag = _HTML_TAGS.get(token.content) if token.type == "html_inline" else None
            if tag:
                closing = token.content.startswith("</")
                token.type = f"{tag}_{'close' if closing else 'open'}"
                token.tag = tag
                token.nesting = -1 if closing else 1
