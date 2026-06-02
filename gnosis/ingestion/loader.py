"""
Document loader for Gnosis.

Supports loading content from:
- Markdown files (.md, .markdown)
- PDF files (.pdf) via PyMuPDF
- Plain text files (.txt)
- Web URLs via httpx + BeautifulSoup

Each loader returns a list of ``Document`` objects carrying the raw text
and source metadata (file path or URL, MIME type, page number where
applicable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class Document:
    """Raw document returned by a loader before chunking."""

    content: str
    source: str
    metadata: dict = field(default_factory=dict)


class Loader(Protocol):
    """Common interface every loader must satisfy."""

    def load(self, source: str) -> list[Document]:
        """Load *source* and return one or more Documents."""
        ...


class MarkdownLoader:
    """Load a single Markdown file into a Document."""

    def load(self, source: str) -> list[Document]:
        """Read *source* and return its content as a Document."""
        ...


class PDFLoader:
    """Load a PDF file page-by-page using PyMuPDF."""

    def load(self, source: str) -> list[Document]:
        """Read each page of *source* and return one Document per page."""
        ...


class TextLoader:
    """Load a plain-text file into a Document."""

    def load(self, source: str) -> list[Document]:
        """Read *source* and return its content as a Document."""
        ...


class URLLoader:
    """Fetch a web URL and return its visible text as a Document."""

    def load(self, source: str) -> list[Document]:
        """Fetch *source*, strip HTML tags, and return a Document."""
        ...


def load(source: str) -> list[Document]:
    """Auto-detect the type of *source* and delegate to the right loader."""
    ...
